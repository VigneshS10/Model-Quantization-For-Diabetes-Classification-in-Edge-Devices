import os
import random
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
import time
import copy
import numpy as np

from resnet import resnet50


def set_random_seeds(random_seed=1000):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def prepare_dataloader(num_workers=8,
                       train_batch_size=32,
                       eval_batch_size=32):

    # train_transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.485, 0.456, 0.406),
    #                          std=(0.229, 0.224, 0.225)),
    # ])

    # test_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.485, 0.456, 0.406),
    #                          std=(0.229, 0.224, 0.225)),
    # ])

    # train_set = torchvision.datasets.CIFAR10(root="data",
    #                                          train=True,
    #                                          download=True,
    #                                          transform=train_transform)
    # We will use test set for validation and test in this project.
    # Do not use test set for validation in practice!
    # test_set = torchvision.datasets.CIFAR10(root="data",
    #                                         train=False,
    #                                         download=True,
    #                                         transform=test_transform)

    # train_sampler = torch.utils.data.RandomSampler(train_set)
    # test_sampler = torch.utils.data.SequentialSampler(test_set)

    # train_loader = torch.utils.data.DataLoader(dataset=train_set,
    #                                            batch_size=train_batch_size,
    #                                            sampler=train_sampler,
    #                                            num_workers=num_workers)

    # test_loader = torch.utils.data.DataLoader(dataset=test_set,
    #                                           batch_size=eval_batch_size,
    #                                           sampler=test_sampler,
    #                                           num_workers=num_workers)

    train_transform_1 = transforms.Compose([transforms.Resize((224)),
                                      transforms.CenterCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.4346, 0.2213, 0.0783),
                                                            std=(0.2439, 0.1311, 0.0703))            
                                    ])
    train_transform_2 = transforms.Compose([transforms.Resize((224)),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.4346, 0.2213, 0.0783),
                                                            std=(0.2439, 0.1311, 0.0703))            
                                    ])

    test_transform = transforms.Compose([transforms.Resize((224)),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.4346, 0.2213, 0.0783),
                                                           std=(0.2439, 0.1311, 0.0703))
                                    ])

    train_set_1 = datasets.ImageFolder('/content/drive/MyDrive/Diabetes_Retina/train', transform=train_transform_1)
    train_set_2 = datasets.ImageFolder('/content/drive/MyDrive/Diabetes_Retina/train', transform=train_transform_2)

    train_set = train_set_1 + train_set_1 + train_set_2 + train_set_2

    test_set   = datasets.ImageFolder('/content/drive/MyDrive/Diabetes_Retina/val', transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(train_set, sampler=train_sampler, batch_size=train_batch_size, num_workers=num_workers)
    test_loader   = torch.utils.data.DataLoader(test_set, sampler=test_sampler,batch_size=eval_batch_size, num_workers=num_workers)

    return train_loader, test_loader


def evaluate_model(model, test_loader, device, criterion=None):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0
    pred = []
    true = []
    for (inputs, labels) in tqdm(test_loader):

        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        pred.append(preds)
        true.append(labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy, pred, true


def train_model(model,
                train_loader,
                test_loader,
                device,
                learning_rate=1e-4,
                num_epochs=200):

    writer = SummaryWriter(log_dir = "/content/drive/MyDrive/Diabetes_Retina/Quant_Resnet_50/events")

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=0.9,
                          weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # Evaluation
    model.eval()
    eval_loss, eval_accuracy, _, _ = evaluate_model(model=model,
                                              test_loader=test_loader,
                                              device=device,
                                              criterion=criterion)

    print('Training Started:')
    test_acc_min = 0
    for epoch in range(num_epochs):

        # Training
        model.train()
        
        running_loss = 0
        running_corrects = 0
        
        print('\n')

        
        for (inputs, labels) in tqdm(train_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

  
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)
        
        
        # Evaluation
        model.eval()
        eval_loss, eval_accuracy, preds, true = evaluate_model(model=model,
                                                  test_loader=test_loader,
                                                  device=device,
                                                  criterion=criterion)

        network_learned = ((100 * eval_accuracy) > test_acc_min)

        if network_learned:
            test_acc_min = (100 * eval_accuracy)
            preds = []
            true = []
            true.append(true)
            preds.append(preds)
            torch.save(model.state_dict(), '/content/drive/MyDrive/Diabetes_Retina/Quant_Resnet_50/models/resnet50.pt')
            print('Improvement-Detected, save-model')

        writer.add_scalar("Train Loss", train_loss, epoch)
        writer.add_scalar("Train Acc", 100.0 * train_accuracy, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Test Loss", eval_loss, epoch)
        writer.add_scalar("Test Acc", 100.0 * eval_accuracy, epoch)
        scheduler.step()

        print(
            "Epoch: {:03d} || Train Loss: {:.3f} || Train Acc: {:.3f} || Eval Loss: {:.3f} || Eval Acc: {:.3f}"
            .format(epoch, train_loss, train_accuracy * 100, eval_loss,
                    eval_accuracy * 100))


    preds = np.array(preds)
    true = np.array(true)

    np.save('/content/drive/MyDrive/Diabetes_Retina/Quant_Resnet_50/predictions/preds.npy', preds)
    np.save('/content/drive/MyDrive/Diabetes_Retina/Quant_Resnet_50/predictions/true.npy', true)

    writer.close()


def calibrate_model(model, loader, device=torch.device("cpu:0")):

    model.to(device)
    model.eval()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)


def measure_inference_latency(model,
                              device,
                              input_size=(1, 3, 224, 224),
                              num_samples=100,
                              num_warmups=10):

    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave


def save_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)


def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model


def save_torchscript_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)


def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)

    return model


def create_model(num_classes=7):

    # The number of channels in ResNet18 is divisible by 8.
    # This is required for fast GEMM integer matrix multiplication.
    # model = torchvision.models.resnet18(pretrained=False)
    model = resnet50(num_classes=1000, pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    # We would use the pretrained ResNet18 as a feature extractor.
    # for param in model.parameters():
    #     param.requires_grad = False

    # Modify the last FC layer
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 10)

    return model


class QuantizedResNet18(nn.Module):
    def __init__(self, model_fp32):
        
        super(QuantizedResNet18, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x


def model_equivalence(model_1,
                      model_2,
                      device,
                      rtol=1e-05,
                      atol=1e-08,
                      num_tests=100,
                      input_size=(1, 3, 224, 224)):

    model_1.to(device)
    model_2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol,
                       equal_nan=False) == False:
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False

    return True


def main():

    random_seed = 0
    num_classes = 7
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model_dir = "/content/drive/MyDrive/Diabetes_Retina/Quant_Resnet_50/models"
    model_filename = "resnet50.pt"
    quantized_model_filename = "resnet50_quantized.pt"
    model_filepath = os.path.join(model_dir, model_filename)
    quantized_model_filepath = os.path.join(model_dir,
                                            quantized_model_filename)

    set_random_seeds(random_seed=random_seed)

    # Create an untrained model.
    model = create_model(num_classes=num_classes)

    train_loader, test_loader = prepare_dataloader(num_workers=8,
                                                   train_batch_size=32,
                                                   eval_batch_size=32)

    # Train model.
    
    # train_model(model=model,
    #             train_loader=train_loader,
    #             test_loader=test_loader,
    #             device=cuda_device,
    #             learning_rate=1e-4,
    #             num_epochs=200)

    model.load_state_dict(torch.load(model_filepath, map_location=cuda_device))
    # # Save model.
    # save_model(model=model, model_dir=model_dir, model_filename=model_filename)
    # Load a pretrained model.
    # model = load_model(model=model,
    #                    model_filepath=model_filepath,
    #                    device=cuda_device)
    # Move the model to CPU since static quantization does not support CUDA currently.
    
    model.to(cpu_device)
    # Make a copy of the model for layer fusion
    fused_model = copy.deepcopy(model)

    model.eval()
    # The model has to be switched to evaluation mode before any layer fusion.
    # Otherwise the quantization will not work correctly.
    fused_model.eval()

    # Fuse the model in place rather manually.
    fused_model = torch.quantization.fuse_modules(fused_model,
                                                  [["conv1", "bn1", "relu"]],
                                                  inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(
                    basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]],
                    inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block,
                                                        [["0", "1"]],
                                                        inplace=True)

    # Print FP32 model.
    print(model)
    # Print fused model.
    print(fused_model)

    # Model and fused model should be equivalent.
    assert model_equivalence(
        model_1=model,
        model_2=fused_model,
        device=cpu_device,
        rtol=1e-03,
        atol=1e-06,
        num_tests=100,
        input_size=(
            1, 3, 224,
            224)), "Fused model is not equivalent to the original model!"

    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    quantized_model = QuantizedResNet18(model_fp32=fused_model)
    # Using un-fused model will fail.
    # Because there is no quantized layer implementation for a single batch normalization layer.
    # quantized_model = QuantizedResNet18(model_fp32=model)
    # Select quantization schemes from
    # https://pytorch.org/docs/stable/quantization-support.html
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    # Custom quantization configurations
    # quantization_config = torch.quantization.default_qconfig
    # quantization_config = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))

    quantized_model.qconfig = quantization_config

    # Print quantization configurations
    print(quantized_model.qconfig)

    # https://pytorch.org/docs/master/torch.quantization.html#torch.quantization.prepare
    torch.quantization.prepare(quantized_model, inplace=True)

    # Use training data for calibration.
    calibrate_model(model=quantized_model,
                    loader=train_loader,
                    device=cpu_device)

    quantized_model = torch.quantization.convert(quantized_model, inplace=True)

    # Using high-level static quantization wrapper
    # The above steps, including torch.quantization.prepare, calibrate_model, and torch.quantization.convert, are also equivalent to
    # quantized_model = torch.quantization.quantize(model=quantized_model, run_fn=calibrate_model, run_args=[train_loader], mapping=None, inplace=False)

    quantized_model.eval()

    # Print quantized model.
    print(quantized_model)

    # Save quantized model.
    save_torchscript_model(model=quantized_model,
                           model_dir=model_dir,
                           model_filename=quantized_model_filename)

    # Load quantized model.
    quantized_jit_model = load_torchscript_model(
        model_filepath=quantized_model_filepath, device=cpu_device)

    _, fp32_eval_accuracy, preds_50, true_50 = evaluate_model(model=model,
                                           test_loader=test_loader,
                                           device=cpu_device,
                                           criterion=None)
    _, int8_eval_accuracy, preds_quant50, true_quant50 = evaluate_model(model=quantized_jit_model,
                                           test_loader=test_loader,
                                           device=cpu_device,
                                           criterion=None)

    preds_50 = np.array(preds_50)
    true_50 = np.array(true_50)
    preds_quant50 = np.array(preds_quant50)
    true_quant50 = np.array(true_quant50)

    np.save('/content/drive/MyDrive/Diabetes_Retina/Quant_Resnet_50/predictions/preds_50.npy', preds_50)
    np.save('/content/drive/MyDrive/Diabetes_Retina/Quant_Resnet_50/predictions/true_50.npy', true_50)
    np.save('/content/drive/MyDrive/Diabetes_Retina/Quant_Resnet_50/predictions/preds_quant50.npy', preds_quant50)
    np.save('/content/drive/MyDrive/Diabetes_Retina/Quant_Resnet_50/predictions/true_quant50.npy', true_quant50)

    # Skip this assertion since the values might deviate a lot.
    # assert model_equivalence(model_1=model, model_2=quantized_jit_model, device=cpu_device, rtol=1e-01, atol=1e-02, num_tests=100, input_size=(1,3,32,32)), "Quantized model deviates from the original model too much!"

    print("Resnet evaluation accuracy: {:.3f}".format(fp32_eval_accuracy * 100))
    print("Quantized Resnet evaluation accuracy: {:.3f}".format(int8_eval_accuracy * 100))

    fp32_cpu_inference_latency = measure_inference_latency(model=model,
                                                           device=cpu_device,
                                                           input_size=(1, 3,
                                                                       224, 224),
                                                           num_samples=100)
    int8_cpu_inference_latency = measure_inference_latency(
        model=quantized_model,
        device=cpu_device,
        input_size=(1, 3, 224, 224),
        num_samples=100)
    int8_jit_cpu_inference_latency = measure_inference_latency(
        model=quantized_jit_model,
        device=cpu_device,
        input_size=(1, 3, 224, 224),
        num_samples=100)
    fp32_gpu_inference_latency = measure_inference_latency(model=model,
                                                           device=cuda_device,
                                                           input_size=(1, 3,
                                                                       224, 224),
                                                           num_samples=100)

    print("Resnet CPU Inference Latency: {:.2f} ms / sample".format(
        fp32_cpu_inference_latency * 1000))
    print("Resnet CUDA Inference Latency: {:.2f} ms / sample".format(
        fp32_gpu_inference_latency * 1000))
    print("Quantized CPU Inference Latency: {:.2f} ms / sample".format(
        int8_cpu_inference_latency * 1000))
    print("Quantized JIT CPU Inference Latency: {:.2f} ms / sample".format(
        int8_jit_cpu_inference_latency * 1000))


if __name__ == "__main__":

    main()
