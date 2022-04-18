import argparse
import torchvision
from torchvision import transforms
import torch
from autoattack import AutoAttack
from robustbench.data import _load_dataset
import models

cifar10_stats = {
    "mean": (0.4914, 0.4822, 0.4465),
    "std": (0.24705882352941178, 0.24352941176470588, 0.2615686274509804),
}

class NormalizedModel(torch.nn.Module):
    
    def __init__(self, model, mean, std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.mean = torch.nn.Parameter(torch.Tensor(mean).view(3, 1, 1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.Tensor(std).view(3, 1, 1), requires_grad=False)
        
    def forward(self, x):
        x = (x - self.mean) / self.std 
        out = self.model(x)
        return out

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--norm", type=str, default="Linf", choices=["Linf", "L2"])
    parser.add_argument("--eps", type=float, default=1/255)
    parser.add_argument("--fast", type=int, default=0, choices=[0, 1])
    
    args = parser.parse_args()
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)

    
    base_model = models.get_model(args.classifier)(in_channels=3, num_classes=10)
    state = torch.load(args.checkpoint, map_location=args.device)
    base_model.load_state_dict(dict((key.replace("model.", ""), value) for (key, value) in
                                     state["state_dict"].items()))
    
    model = NormalizedModel(base_model, cifar10_stats["mean"], cifar10_stats["std"]).to(args.device)
    model.eval()

    x_test, y_test = _load_dataset(testset)
    if args.fast:
        adversary = AutoAttack(model, norm=args.norm, eps=args.eps, device=args.device, version="custom", attacks_to_run=["apgd-ce"])
    else:    
        adversary = AutoAttack(model, norm=args.norm, eps=args.eps, device=args.device)
    x_adv = adversary.run_standard_evaluation(x_test, y_test)

if __name__ == "__main__":
    main()
