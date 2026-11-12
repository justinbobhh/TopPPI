import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None):
        device = (features.device if features.is_cuda else torch.device('cpu'))
        batch_size = features.shape[0]

        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_feature = features
        anchor_feature = features
        anchor_count = 1

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1.0, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss

class DualBranchLoss_MC(nn.Module):

    def __init__(self, alpha=0.1, temperature=0.07):
        super().__init__()
        self.alpha = alpha

        self.loss_main = nn.CrossEntropyLoss()
        self.loss_supcon = SupConLoss(temperature=temperature)

    def forward(self, logits, proj_features, labels):

        l_main = self.loss_main(logits, labels)

        l_supcon = self.loss_supcon(proj_features, labels)

        total_loss = l_main + self.alpha * l_supcon
        return total_loss, l_main, l_supcon
