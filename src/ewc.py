import torch
import torch.nn as nn
import torch.autograd as autograd

class EWC:
    def __init__(self, model):
        self.model = model
        self.fisher_information = {}
        self.previous_parameters = {}

    def calculate_fisher(self, data_loader, device):
        """
        Calculate the Fisher Information Matrix for the model parameters.
        """
        self.model.eval()
        fisher = {name: torch.zeros_like(param, device=device) for name, param in self.model.named_parameters()}

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            self.model.zero_grad()

            # Compute outputs and loss
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()

            # Accumulate Fisher Information
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.pow(2).detach()

        # Normalize Fisher Information Matrix
        for name in fisher:
            fisher[name] /= len(data_loader)
        self.fisher_information = fisher

    def update(self, model):
        """
        Store the current parameters of the model.
        """
        self.previous_parameters = {name: param.clone().detach() for name, param in model.named_parameters()}

    def penalty(self, model):
        """
        Calculate the EWC penalty for the model.
        """
        penalty = 0
        for name, param in model.named_parameters():
            if name in self.previous_parameters:
                fisher = self.fisher_information.get(name, torch.zeros_like(param))
                penalty += (fisher * (param - self.previous_parameters[name]).pow(2)).sum()
        return penalty
