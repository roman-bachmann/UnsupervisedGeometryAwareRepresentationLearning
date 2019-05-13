import torch

class LossOnDict(torch.nn.Module):
    def __init__(self, key, loss):
        super(LossOnDict, self).__init__()
        self.key = key
        self.loss = loss

    def forward(self, pred_dict, label_dict):
        return self.loss(pred_dict[self.key], label_dict[self.key])

class KLLoss(torch.nn.Module):
    def __init__(self, mu_key, logvar_key, reduction='mean'):
        super(KLLoss, self).__init__()
        self.mu_key = mu_key
        self.logvar_key = logvar_key
        self.reduction = reduction

    def forward(self, pred_dict, label_dict):
        mu = pred_dict[self.mu_key]
        logvar = pred_dict[self.logvar_key]
        if self.reduction == 'none':
            return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        elif self.reduction == 'mean':
            return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        elif self.reduction == 'sum':
            return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            raise Exception('No valid reduction specified.')


class PreApplyCriterionListDict(torch.nn.Module):
    """
    Wraps a loss operating on tensors into one that processes dict of labels and predictions
    """
    def __init__(self, criterions_single, sum_losses=True, loss_weights=None):
        super(PreApplyCriterionListDict, self).__init__()
        self.criterions_single = criterions_single
        self.sum_losses = sum_losses
        self.loss_weights = loss_weights

    def forward(self, pred_dict, label_dict, iteration=None):
        """
        The loss is computed as the sum of all the loss values
        :param pred_dict: List containing the predictions
        :param label_dict: List containing the labels
        :return: The sum of all the loss values computed
        """
        losslist = []
        for criterion_idx, criterion_single in enumerate(self.criterions_single):
            loss_i = criterion_single(pred_dict, label_dict)
            if self.loss_weights is not None:
                loss_i = loss_i * self.loss_weights[criterion_idx]
            losslist.append(loss_i)

        if self.sum_losses:
            return sum(losslist)
        else:
            return losslist

class PreApplyCriterionDictDict(torch.nn.Module):
    """
    Wraps a loss operating on tensors into one that processes dict of labels and predictions.
    Includes KL annealing.
    """
    def __init__(self, criterions_single, sum_losses=True, loss_weights=None, KL_annealing=0):
        super(PreApplyCriterionDictDict, self).__init__()
        self.criterions_single = criterions_single
        self.sum_losses = sum_losses
        self.loss_weights = loss_weights
        self.KL_annealing = KL_annealing

    def forward(self, pred_dict, label_dict, iteration=-1):
        """
        The loss is computed as the sum of all the loss values
        :param pred_dict: List containing the predictions
        :param label_dict: List containing the labels
        :return: The sum of all the loss values computed
        """
        annealing_factor = 1 if iteration < 0 or iteration >= self.KL_annealing else iteration / self.KL_annealing

        losslist = []
        for criterion_id, criterion_single in self.criterions_single.items():
            loss_i = criterion_single(pred_dict, label_dict)
            if self.loss_weights is not None:
                loss_i = loss_i * self.loss_weights[criterion_id]
            if self.KL_annealing > 0 and (criterion_id == 'kl_fg' or criterion_id == 'kl_3d'):
                loss_i = loss_i * annealing_factor
            losslist.append(loss_i)

        if self.sum_losses:
            return sum(losslist)
        else:
            return losslist

class LossLabelMeanStdNormalized(torch.nn.Module):
    """
    Normalize the label before applying the specified loss (could be normalized loss..)
    """
    def __init__(self, key, loss_single, subjects=False, weight=1):
        super(LossLabelMeanStdNormalized, self).__init__()
        self.key = key
        self.loss_single = loss_single
        self.subjects = subjects
        self.weight=weight

    def forward(self, preds, labels):
        pred_pose = preds[self.key]
        label_pose = labels[self.key]
        label_mean = labels['pose_mean']
        label_std = labels['pose_std']
        label_pose_norm = (label_pose-label_mean)/label_std

        if self.subjects:
            info = labels['frame_info']
            subject = info.data.cpu()[:,3]
            errors = [self.loss_single.forward(pred_pose[i], label_pose_norm[i]) for i,x in enumerate(pred_pose) if subject[i] in self.subjects]
            #print('subject',subject,'errors',errors)
            if len(errors) == 0:
                return torch.autograd.Variable(torch.FloatTensor([0])).cuda()
            return self.weight * sum(errors) / len(errors)

        return self.weight * self.loss_single.forward(pred_pose,label_pose_norm)

class LossLabelMeanStdUnNormalized(torch.nn.Module):
    """
    UnNormalize the prediction before applying the specified loss (could be normalized loss..)
    """
    def __init__(self, key, loss_single, scale_normalized=False, weight=1):
        super(LossLabelMeanStdUnNormalized, self).__init__()
        self.key = key
        self.loss_single = loss_single
        self.scale_normalized = scale_normalized
        #self.subjects = subjects
        self.weight=weight

    def forward(self, preds, labels):
        label_pose = labels[self.key]
        label_mean = labels['pose_mean']
        label_std = labels['pose_std']
        pred_pose = preds[self.key]

        if self.scale_normalized:
            per_frame_norm_label = label_pose.norm(dim=1).expand_as(label_pose)
            per_frame_norm_pred  = pred_pose.norm(dim=1).expand_as(label_pose)
            pred_pose = pred_pose / per_frame_norm_pred * per_frame_norm_label

        pred_pose_norm = (pred_pose*label_std) + label_mean

        return self.weight*self.loss_single.forward(pred_pose_norm, label_pose)
