{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyMbfZAmV6mJ57KEP+FPMBKy"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "OxQBg7a4d7wo"
      },
      "outputs": [],
      "source": [
        "# Install ESRGAN dependencies : Python 3, PyTorch>=1.0, CUDA>=7.5, OpenCV"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install torch torchvision"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "z-dkAsd4elzi",
        "outputId": "af277477-2100-444c-e0a3-5e4420d6804e"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (2.1.0+cu118)\n",
            "Requirement already satisfied: torchvision in /usr/local/lib/python3.10/dist-packages (0.16.0+cu118)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch) (3.13.1)\n",
            "Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch) (4.5.0)\n",
            "Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch) (1.12)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch) (3.2.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch) (3.1.2)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch) (2023.6.0)\n",
            "Requirement already satisfied: triton==2.1.0 in /usr/local/lib/python3.10/dist-packages (from torch) (2.1.0)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from torchvision) (1.23.5)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from torchvision) (2.31.0)\n",
            "Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.10/dist-packages (from torchvision) (9.4.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch) (2.1.3)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision) (3.6)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision) (2023.11.17)\n",
            "Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch) (1.3.0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!nvidia-smi"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "s2_o857FeVke",
        "outputId": "aa33cb3d-63cf-4838-cf5c-a47e21f54c00"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Sat Dec  2 21:42:15 2023       \n",
            "+-----------------------------------------------------------------------------+\n",
            "| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |\n",
            "|-------------------------------+----------------------+----------------------+\n",
            "| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |\n",
            "| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |\n",
            "|                               |                      |               MIG M. |\n",
            "|===============================+======================+======================|\n",
            "|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |\n",
            "| N/A   51C    P8    10W /  70W |      0MiB / 15360MiB |      0%      Default |\n",
            "|                               |                      |                  N/A |\n",
            "+-------------------------------+----------------------+----------------------+\n",
            "                                                                               \n",
            "+-----------------------------------------------------------------------------+\n",
            "| Processes:                                                                  |\n",
            "|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |\n",
            "|        ID   ID                                                   Usage      |\n",
            "|=============================================================================|\n",
            "|  No running processes found                                                 |\n",
            "+-----------------------------------------------------------------------------+\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install numpy opencv-python"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "P8l8JYghe21l",
        "outputId": "abf8728b-87fb-417f-b200-dc2feb63143d"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (1.23.5)\n",
            "Requirement already satisfied: opencv-python in /usr/local/lib/python3.10/dist-packages (4.8.0.76)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "import numpy as np\n",
        "import cv2\n",
        "\n",
        "print(\"PyTorch Version:\", torch.__version__)\n",
        "print(\"CUDA Available:\", torch.cuda.is_available())\n",
        "print(\"NumPy Version:\", np.__version__)\n",
        "print(\"OpenCV Version:\", cv2.__version__)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mlSOkZ9MfH4Q",
        "outputId": "fbccdbb1-4e5b-47c8-9dfb-53751de08619"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "PyTorch Version: 2.1.0+cu118\n",
            "CUDA Available: True\n",
            "NumPy Version: 1.23.5\n",
            "OpenCV Version: 4.8.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!git clone https://github.com/xinntao/ESRGAN"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "N2gy87mcfSm-",
        "outputId": "913ee135-650a-421d-9abe-8ca48dbd603e"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Cloning into 'ESRGAN'...\n",
            "remote: Enumerating objects: 225, done.\u001b[K\n",
            "remote: Total 225 (delta 0), reused 0 (delta 0), pack-reused 225\u001b[K\n",
            "Receiving objects: 100% (225/225), 24.86 MiB | 18.66 MiB/s, done.\n",
            "Resolving deltas: 100% (86/86), done.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!cd ESRGAN"
      ],
      "metadata": {
        "id": "WY_B0ezifYtl"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# pretrained model \"RRDB_ESRGAN_x4.pth\" uploaded to ./ESRGAN/models"
      ],
      "metadata": {
        "id": "iC4vS0SC75FS"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "D2Zb1F_L8DcY"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
