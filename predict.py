import os
import torch
import torch.nn as nn
import numpy as np
import cv2

from configs.config import Config
from models.SimpleNet import SimpleNet
from models.VGGNet import VGGNet
from models.ResNet import ResNet
from models.MobileNet import MobileNet

from src.dataset import get_loaders
from src.utils import calculate_all_metrics
from src.visualizer import save_confusion_matrix
from src.gradcam import GradCam


def build_model(model_name, num_classes=2):
    """
    Config.MODEL_NAME에 따라 사용할 모델을 생성하는 함수.
    """

    if model_name == "SimpleNet":
        return SimpleNet(in_channels=3, num_classes=num_classes)

    elif model_name == "VGGNet":
        return VGGNet(in_channels=3, num_classes=num_classes)

    elif model_name == "ResNet":
        return ResNet(in_channels=3, num_classes=num_classes)

    elif model_name == "MobileNet":
        return MobileNet(in_channels=3, num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model name: {model_name}")


def find_last_conv2d(module):

    last_conv = None

    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            last_conv = layer

    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")

    return last_conv


def get_gradcam_target_layer(model, model_name):

    if model_name == "SimpleNet":
        # SimpleNet의 마지막 convolution block
        return model.block3

    elif model_name == "VGGNet":
        # VGGNet의 마지막 Conv2d layer
        return find_last_conv2d(model.features)

    elif model_name == "ResNet":
        # ResNet의 마지막 residual block
        return model.stage4[-1]

    elif model_name == "MobileNet":
        # MobileNet의 마지막 depthwise separable convolution block
        return model.features[-1]

    else:
        raise ValueError(f"Unknown model name for Grad-CAM: {model_name}")


def load_model_weights(model, model_path, device):

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        state_dict = torch.load(
            model_path,
            map_location=device
        )

    model.load_state_dict(state_dict)

    return model


def tensor_to_uint8_bgr(image_tensor):

    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()

    # 시각화용 min-max normalization
    image = image - image.min()
    image = image / (image.max() + 1e-7)

    image = np.uint8(255 * image)

    # OpenCV는 BGR 순서를 사용함
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def save_gradcam_overlay(image_tensor, mask, save_path):
    """
    원본 이미지와 Grad-CAM heatmap을 합성하여 저장.
    """

    orig_img = tensor_to_uint8_bgr(image_tensor)

    heatmap = cv2.applyColorMap(
        np.uint8(255 * mask),
        cv2.COLORMAP_JET
    )

    cam_result = cv2.addWeighted(
        orig_img,
        0.6,
        heatmap,
        0.4,
        0
    )

    cv2.imwrite(save_path, cam_result)


def predict():
    device = Config.device
    model_name = Config.MODEL_NAME

    model_path = os.path.join(
        Config.save_dir,
        model_name,
        "best_model.pth"
    )

    save_dir = os.path.join(
        Config.save_dir,
        model_name
    )

    gradcam_save_dir = os.path.join(
        save_dir,
        "gradcam_results"
    )

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(gradcam_save_dir, exist_ok=True)

    _, _, test_loader = get_loaders(
        Config.data_dir,
        Config.batch_size,
        return_test_paths=True
    )

    model = build_model(
        model_name=model_name,
        num_classes=2
    ).to(device)

    model = load_model_weights(
        model=model,
        model_path=model_path,
        device=device
    )

    print(f"Model loaded: {model_path}")

    model.eval()

    target_layer = get_gradcam_target_layer(
        model=model,
        model_name=model_name
    )

    gcam = GradCam(
        model=model,
        target_layer=target_layer
    )

    all_outputs = []
    all_labels = []

    print(f"Testing & Generating Grad-CAM for {model_name}...")

    # batch size x gradcam_batch_limit

    gradcam_batch_limit = 10

    for i, (images, labels, paths) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        if i < gradcam_batch_limit:
            for j in range(images.size(0)):
                img_tensor = images[j].unsqueeze(0)

                # 이미지에 대한 예측값과 확률 계산
                with torch.no_grad():
                    single_output = model(img_tensor)
                    single_prob = torch.softmax(single_output, dim=1)[0]
                    pred_idx = single_prob.argmax().item()
                    pred_prob = single_prob[pred_idx].item()

                mask = gcam(
                    img_tensor,
                    target_index=pred_idx
                )

                gt_idx = labels[j].item()

                gt_name = "PNEUMONIA" if gt_idx == 1 else "NORMAL"
                pred_name = "PNEUMONIA" if pred_idx == 1 else "NORMAL"

                original_path = paths[j]
                original_name = os.path.basename(original_path)
                original_name = os.path.splitext(original_name)[0]

                save_filename = (
                    f"{i:03d}_{j:02d}_"
                    f"{original_name}_"
                    f"GT-{gt_name}_"
                    f"Pred-{pred_name}_"
                    f"Prob-{pred_prob:.2f}.png"
                )

                save_path = os.path.join(
                    gradcam_save_dir,
                    save_filename
                )

                save_gradcam_overlay(
                    image_tensor=images[j],
                    mask=mask,
                    save_path=save_path
                )

        with torch.no_grad():
            outputs = model(images)
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    final_outputs = torch.cat(all_outputs, dim=0)
    final_labels = torch.cat(all_labels, dim=0)

    _, final_preds = torch.max(final_outputs, 1)

    metrics = calculate_all_metrics(
        final_outputs,
        final_labels
    )

    print("\nTest Metrics")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print(f"\nGrad-CAM results saved to: {gradcam_save_dir}")

    save_confusion_matrix(
        final_labels.numpy(),
        final_preds.numpy(),
        ["NORMAL", "PNEUMONIA"],
        save_dir
    )

    gcam.remove_hooks()


if __name__ == "__main__":
    predict()