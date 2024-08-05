import argparse
import json
import torch
from torchvision import models
from PIL import Image
import numpy as np

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K predictions')
    parser.add_argument('--category_names', type=str, help='Path to category to name JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif checkpoint['arch'] == 'resnet18':
        model = models.resnet18(pretrained=True)
    else:
        raise ValueError("Unsupported architecture")

    model.classifier = nn.Sequential(
        nn.Linear(checkpoint['input_size'], checkpoint['hidden_units']),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(checkpoint['hidden_units'], 102),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = preprocess(image).numpy()
    return image

def predict(image_path, model, topk=1, device='cpu'):
    model.eval()
    model.to(device)
    
    image = process_image(image_path)
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model.forward(image)
    
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)
    return top_p.cpu().numpy()[0], top_class.cpu().numpy()[0]

def main():
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    model = load_checkpoint(args.checkpoint)
    model.to(device)
    
    top_p, top_class = predict(args.input, model, args.top_k, device)
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_class = [cat_to_name[str(c)] for c in top_class]
    
    print(f"Top {args.top_k} classes: {top_class}")
    print(f"Probabilities: {top_p}")

if __name__ == '__main__':
    main()
