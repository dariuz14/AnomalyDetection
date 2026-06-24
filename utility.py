import torch
import numpy as np
import os

def save_attention_maps(model, data_loader, path, device):
    
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, images in enumerate(data_loader):
            images = images.to(device)
            _, cls_attn = model(images)

            cls_attn = np.squeeze(cls_attn, axis=1)
            for i in range(cls_attn.shape[0]):
                
                attn = cls_attn[i]
                
                p_low, p_high = np.percentile(attn, [2, 98])
                
                attn_clipped = np.clip(attn, p_low, p_high)
                attn_norm = (attn_clipped - p_low) / (p_high - p_low)

                save_path = os.path.join(path, f"attn_{batch_idx}_{i}.npy")
                np.save(save_path, attn_norm) # id_train / id_test / ood

def extract_max_softmax(model, data_loader, device):

    model = model.to(device)
    model.eval()

    all_scores = []

    with torch.no_grad():
        for images in data_loader:
            images = images.to(device)

            logits, _ = model(images)

            probs = torch.nn.functional.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            all_scores.extend(max_probs.cpu().numpy())
    
    return np.array(all_scores)

                
if __name__ == "__main__":
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    
    from dataset import CustomDataset
    from model import load_model
    from config import train_id_data_path, test_id_data_path, ood_data_path 
    

    my_transform = transforms.Compose([
        transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ID train dataset
    #id_train_dataset = ImageFolder(train_id_data_path, transform=my_transform)
    #id_train_loader = DataLoader(id_train_dataset, batch_size=4, shuffle=False)

    # ID test dataset
    #id_test_dataset = ImageFolder(test_id_data_path, transform=my_transform)
    #id_test_loader = DataLoader(id_test_dataset, batch_size=4, shuffle=False)
    
    # OOD dataset
    ood_dataset = CustomDataset(ood_data_path, transform=my_transform)
    ood_loader = DataLoader(ood_dataset, batch_size=4, shuffle=False)
    
    from config import attention_maps_id_train_path, attention_maps_id_test_path, attention_maps_ood_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    my_model = load_model(extract_attention=True)

    #save_attention_maps(model=my_model, data_loader=ood_loader, path=attention_maps_ood_path, device=device)

    scores = extract_max_softmax(model=my_model, data_loader=ood_loader, device=device)
    np.save("ood_maxsoftmax", scores)