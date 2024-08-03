
import torch
from torchvision.transforms import v2
from datasets import load_dataset
from torch.utils.data import DataLoader

def _reshape(x):
    result = 1
    for i in range(len(x.shape)):
        result *= x.shape[i]
    return x.reshape(result)


def preprocess_hugging_face_image(path, transforms, num_classes, one_hot=False, split='train'):
    ds = load_dataset(path, "en-US", split=split)
    dataset = ds.train_test_split(test_size=0.1)
    training_data = dataset['train']
    test_data = dataset['test']


    def trfs(examples):
        features = list(training_data.features.keys())
        x_key = features[0]
        y_key = features[1]
        examples[x_key] = [_reshape(transforms(x.convert("RGB"))) for x in examples[x_key]]
        if one_hot:
            examples[y_key] = [torch.nn.functional.one_hot(torch.tensor(label),num_classes) for label in examples[y_key]]
        else:
            examples[y_key] = [torch.tensor(label)for label in examples[y_key]]         
        return examples

    training_data.set_transform(trfs)
    test_data.set_transform(trfs)
    train_dataloader = DataLoader(training_data, batch_size=100)
    test_dataloader = DataLoader(test_data, batch_size=100)
    return train_dataloader, test_dataloader