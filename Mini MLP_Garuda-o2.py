{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMut0L2P7Z+oUnC4RF/3x24",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Fauzan22-dev/Pasukan_AI2030/blob/main/Mini%20MLP_Garuda-o2.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "9P6JjfQukzzT",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f70fa0f2-551b-4e36-ef9d-9762921a37ad"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 0 - Loss: 0.7751\n",
            "Prediksi akhir:\n",
            " [[0.991198   0.008802  ]\n",
            " [0.98711881 0.01288119]\n",
            " [0.00740909 0.99259091]\n",
            " [0.04245491 0.95754509]]\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "\n",
        "def relu(x):\n",
        "    return np.maximum(0, x)\n",
        "\n",
        "def relu_derivative(x):\n",
        "    return (x > 0).astype(float)\n",
        "\n",
        "def softmax(x):\n",
        "    exp = np.exp(x - np.max(x, axis=1, keepdims=True))\n",
        "    return exp / np.sum(exp, axis=1, keepdims=True)\n",
        "\n",
        "def cross_entropy(pred, true):\n",
        "    return -np.sum(true * np.log(pred + 1e-9)) / pred.shape[0]\n",
        "\n",
        "X = np.array([\n",
        "    [1, 0],\n",
        "    [0, 1],\n",
        "    [1, 1],\n",
        "    [0, 0]\n",
        "])\n",
        "\n",
        "y_true = np.array([\n",
        "    [1, 0],\n",
        "    [1, 0],\n",
        "    [0, 1],\n",
        "    [0, 1]\n",
        "])\n",
        "\n",
        "np.random.seed(0)\n",
        "w1 = np.random.randn(2, 4)\n",
        "b1 = np.zeros((1, 4))\n",
        "w2 = np.random.randn(4, 2)\n",
        "b2 = np.zeros((1, 2))\n",
        "lr = 0.1\n",
        "\n",
        "for epoch in range(1000):\n",
        "    z1 = np.dot(X, w1) + b1\n",
        "    a1 = relu(z1)\n",
        "    z2 = np.dot(a1, w2) + b2\n",
        "    a2 = softmax(z2)\n",
        "    loss = cross_entropy(a2, y_true)\n",
        "\n",
        "    dz2 = a2 - y_true\n",
        "    dw2 = np.dot(a1.T, dz2) / X.shape[0]\n",
        "    db2 = np.sum(dz2, axis=0, keepdims=True) / X.shape[0]\n",
        "\n",
        "    da1 = np.dot(dz2, w2.T)\n",
        "    dz1 = da1 * relu_derivative(z1)\n",
        "    dw1 = np.dot(X.T, dz1) / X.shape[0]\n",
        "    db1 = np.sum(dz1, axis=0, keepdims=True) / X.shape[0]\n",
        "\n",
        "    w2 -= lr * dw2\n",
        "    b2 -= lr * db2\n",
        "    w1 -= lr * dw1\n",
        "    b1 -= lr * db1\n",
        "\n",
        "    if epoch % 100 == 0:\n",
        "        print(f\"Epoch {epoch} - Loss: {loss:.4f}\")\n",
        "\n",
        "z1 = np.dot(X, w1) + b1\n",
        "a1 = relu(z1)\n",
        "z2 = np.dot(a1, w2) + b2\n",
        "a2 = softmax(z2)\n",
        "print(\"Prediksi akhir:\\n\", a2)"
      ]
    }
  ]
}