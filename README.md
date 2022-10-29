<div align="center">
  
# Model Quantization to optimize Alexnet, MobileNet and Resnet-50 for Diabetes Classification
</div align="center"> 

Quantization for deep learning is the process of approximating a neural network that uses floating-point numbers by a neural network of low bit width numbers. This dramatically reduces both the memory requirement and computational cost of using neural networks.

In this repository, I have implemented **static quantization** strategy for Alexnet, MobileNet and Resnet-50 trained on a Retina based Diabetes Classification dataset and also have proven the benefits occuring from doing so by giving relevant statistics and evaluation metrics.

---

  
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Dataset
Download the dataset from the links given below.

[Train](https://drive.google.com/drive/folders/1LI9RDJRTOKUKfwC_dXKMBwHa4Y-OA6eX?usp=share_link)\
[Train2](https://drive.google.com/drive/folders/1--4A1O_T2FdijAa48877YyIxZ-Mt5HtK?usp=share_link)\
[Val](https://drive.google.com/drive/folders/144XCsP-3U1ld0SUIXp568o-9Ie6I7Dnc?usp=share_link)

## Training and Evaluation

1. Download the dataset from the link above.
2. Be sure to change the file directory paths in all the required places as per your convenience.
2. Run the quant python files to train your model on the diabetes classification dataset.
3. Both the normal model and the quantized models will be saved in the events folder, the predictions in the predictions folder and event logs in the event folder.
4. Statistics such as Accuracy, Model size and CPU inference latency are all in built for evaluation in the quant python files.

## Results

From the statistics given below, my quantization strategy reduced the model size, and the inference latency multiple folds while maintaining the accuracy of the model without any noticeble decline.

|            Model            |Top 1 Acc (%)| Model size| CPU Inference Latency |
|-----------------------------|-----------  |-----------|-----------------------|
| Alexnet                     |   42.537    |90 MB      |37.94 ms/sample        |
| **Quantized Alexnet**       |  **42.537** |**23.2 MB**| **29.65 ms/sample**   |

---

|            Model            |Top 1 Acc (%)| Model size| CPU Inference Latency |
|-----------------------------|-----------  |-----------|-----------------------|
| MobileNet                   |   58.46     |8.7 MB     |10.23 ms/sample        |
| **Quantized MobileNet**     |  **58.41**  |**2.8 MB** | **6.27 ms/sample**    |

---

|            Model            |Top 1 Acc (%)| Model size| CPU Inference Latency |
|-----------------------------|-----------  |-----------|-----------------------|
| Resnet50                    |   62.44     |90 MB      |162.19 ms/sample       |
| **Quantized Resnet50**      |  **63.41**  |**23.2 MB**| **101.07 ms/sample**  |

## Pretrained models
Pretrained models for the all the mentioned models can be downloaded from this [link](https://drive.google.com/drive/folders/1cyvzv0cl4PxqV_DTJZf0EwkBZvwEJFEr?usp=share_link) for inference.

## Contact
For any queries, feel free to contact at vignesh.nitt10@gmail.com.
