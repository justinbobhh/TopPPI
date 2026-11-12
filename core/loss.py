import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        labels = labels.view(-1, 1)
        label_mask = torch.eq(labels, labels.T).float()

        logits = similarity_matrix / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * (1 - mask.float())
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = label_mask * (1 - mask.float())

        if mask_pos_pairs.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask_pos_pairs * log_prob).sum(1) / (mask_pos_pairs.sum(1) + 1e-6)
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss

class DualBranchLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=1.0, temperature=0.07, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.loss_main = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.loss_supcon = SupConLoss(temperature=temperature)

    def forward(self, logits, proj_features, labels):

        l_main = self.loss_main(logits.view(-1), labels.float())

        l_supcon = self.loss_supcon(proj_features, labels)

        total_loss = self.beta * l_main + self.alpha * l_supcon
        return total_loss, l_main, l_supcon
