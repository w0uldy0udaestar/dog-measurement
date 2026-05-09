"""
BITE 추론 파이프라인 (gradio_demo.py에서 핵심 로직만 추출).

gradio, dominate, distutils 등 불필요한 의존성 제거.
Windows 네이티브 + Python 3.10 + PyTorch 2.x 호환.

사용법:
    runner = BITERunner("path/to/bite_release")
    result = runner.run("path/to/dog_image.jpg")
    # result["vertices"]: (3889, 3) numpy array
    # result["faces"]: (7774, 3) numpy array
    # result["betas"]: (30,) shape parameters
"""

import json
import os
import pickle as pkl
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchvision
import trimesh
from PIL import Image


class BITERunner:
    """BITE 추론을 실행하는 자체 파이프라인."""

    def __init__(self, bite_dir: str, device: str = "auto",
                 apply_ttopt: bool = True, ttopt_steps: int = 301):
        self.bite_dir = Path(bite_dir)
        self.apply_ttopt = apply_ttopt
        self.ttopt_steps = ttopt_steps

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._add_bite_to_path()
        self._load_models()

    def _add_bite_to_path(self):
        """BITE 디렉토리를 sys.path 최상단에 등록."""
        bite_str = str(self.bite_dir)
        if bite_str not in sys.path:
            sys.path.insert(0, bite_str)

    def _load_models(self):
        print(f"[BITERunner] Loading models on {self.device}...")

        # run_spike.py가 importlib로 이 파일을 직접 로드하므로
        # 우리 프로젝트의 src 패키지가 sys.modules에 없음.
        # BITE의 src가 정상적으로 import됨.
        from src.configs.defaults import get_cfg_defaults
        from src.configs.defaults_global import get_cfg_global_updated, update_cfg_global_with_yaml
        from src.combined_model.bite_inference_model_for_ttopt import BITEInferenceModel
        from src.smal_pytorch.smal_model.smal_torch_new import SMAL
        from src.stacked_hourglass.utils.imutils import get_norm_dict
        from src.smal_pytorch.renderer.silh_renderer import SilhRenderer
        from src.configs.SMAL_configs import SMAL_MODEL_CONFIG

        config_name = "refinement_cfg_test_withvertexwisegc_csaddnonflat.yaml"
        checkpoint_name = "cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0_forrelease_v0/checkpoint.pth.tar"

        path_config = str(self.bite_dir / "src" / "configs" / config_name)
        update_cfg_global_with_yaml(path_config)
        self.cfg = get_cfg_global_updated()

        path_checkpoint = str(self.bite_dir / "checkpoint" / checkpoint_name)
        if not os.path.exists(path_checkpoint) or os.path.getsize(path_checkpoint) < 1000:
            raise FileNotFoundError(
                f"체크포인트가 없거나 LFS 포인터입니다: {path_checkpoint}\n"
                f"git lfs pull --include='checkpoint/**' 을 실행하세요."
            )

        norm_dict = get_norm_dict(data_info=None, device=self.device)
        self.bite_model = BITEInferenceModel(self.cfg, path_checkpoint, norm_dict)

        smal_model_type = self.bite_model.smal_model_type
        logscale_part_list = SMAL_MODEL_CONFIG[smal_model_type]["logscale_part_list"]
        self.smal = SMAL(
            smal_model_type=smal_model_type,
            template_name="neutral",
            logscale_part_list=logscale_part_list,
        ).to(self.device)
        self.silh_renderer = SilhRenderer(image_size=256).to(self.device)

        self.bbox_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        ).to(self.device)
        self.bbox_model.eval()

        loss_weight_path = str(
            self.bite_dir / "src" / "configs" / "ttopt_loss_weights" / "bite_loss_weights_ttopt.json"
        )
        with open(loss_weight_path) as f:
            self.loss_weights = json.loads(f.read())

        remeshing_path = str(
            self.bite_dir / "data" / "smal_data_remeshed"
            / "uniform_surface_sampling" / "my_smpl_39dogsnorm_Jr_4_dog_remesh4000_info.pkl"
        )
        with open(remeshing_path, "rb") as fp:
            remeshing_dict = pkl.load(fp)
        self.remeshing_faces = torch.tensor(
            remeshing_dict["smal_faces"][remeshing_dict["faceid_closest"]],
            dtype=torch.long, device=self.device,
        )
        self.remeshing_barys = torch.tensor(
            remeshing_dict["barys_closest"],
            dtype=torch.float32, device=self.device,
        )

        self.SMAL_MODEL_CONFIG = SMAL_MODEL_CONFIG
        print(f"[BITERunner] Models loaded.")

    def detect_bbox(self, image_path: str) -> Optional[list]:
        """FasterRCNN으로 강아지 bounding box 검출."""
        img = Image.open(image_path).convert("RGB")
        img_tensor = torchvision.transforms.functional.to_tensor(img).to(self.device)

        with torch.no_grad():
            predictions = self.bbox_model([img_tensor])

        pred = predictions[0]
        dog_indices = [
            i for i, label in enumerate(pred["labels"])
            if label == 18 and pred["scores"][i] > 0.5
        ]

        if not dog_indices:
            print(f"[WARN] 강아지가 감지되지 않음: {image_path}")
            return None

        best = dog_indices[0]
        box = pred["boxes"][best].cpu().numpy()
        return [(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))]

    def run(self, image_path: str, bbox: list = None) -> Dict:
        """
        단일 이미지에서 BITE 추론 실행.

        Returns:
            dict with keys:
              - vertices: (3889, 3) SMAL 메시 vertex 좌표 (변환 없는 원본)
              - faces: (7774, 3) face indices
              - betas: (30,) shape parameters
              - betas_limbs: (7,) limb shape parameters
              - pose: pose parameters
              - image_path: 입력 이미지 경로
        """
        from src.stacked_hourglass.utils.imutils import get_norm_dict
        from src.combined_model.loss.loss_utils import reset_loss_values
        from src.smal_pytorch.utils import get_optimed_pose_with_glob
        from src.smal_pytorch.utils import rotmat_to_rot6d
        from src.combined_model.loss.ttopt_loss_utils import (
            leg_sideway_error, leg_torsion_error,
            tail_sideway_error, tail_torsion_error,
            spine_sideway_error, spine_torsion_error,
            calculate_plane_errors_batch,
        )
        from src.combined_model.datasets.stanext24_withgc import StanExtGC as StanExt
        from src.stacked_hourglass.datasets.samplers.custom_crop_from_image import get_single_crop_dataset_from_image

        if bbox is None:
            bbox = self.detect_bbox(image_path)

        val_dataset, val_loader, len_val, test_name_list, data_info, acc_joints = \
            get_single_crop_dataset_from_image(image_path, bbox=bbox)

        norm_dict = get_norm_dict(data_info, self.device)
        keypoint_weights = torch.tensor(
            data_info.keypoint_weights, dtype=torch.float
        )[None, :].to(self.device)

        for i, (input_batch, target_dict) in enumerate(val_loader):
            for key in target_dict:
                if key == "breed_index":
                    target_dict[key] = target_dict[key].long().to(self.device)
                elif key in ["index", "pts", "tpts", "target_weight", "silh",
                             "silh_distmat_tofg", "silh_distmat_tobg",
                             "sim_breed_index", "img_border_mask"]:
                    target_dict[key] = target_dict[key].float().to(self.device)
                elif key == "has_seg":
                    target_dict[key] = target_dict[key].to(self.device)
            input_batch = input_batch.float().to(self.device)
            break

        preds_dict = self.bite_model.get_all_results(input_batch)
        res = self.bite_model.get_selected_results(
            preds_dict=preds_dict, result_networks=["ref"]
        )["ref"]

        bs = res["pose_rotmat"].shape[0]
        all_pose_6d = rotmat_to_rot6d(
            res["pose_rotmat"][:, None, 1:, :, :].clone().reshape((-1, 3, 3))
        ).reshape((bs, -1, 6))
        all_orient_6d = rotmat_to_rot6d(
            res["pose_rotmat"][:, None, :1, :, :].clone().reshape((-1, 3, 3))
        ).reshape((bs, -1, 6))

        ind_img = 0
        optimed_pose_6d = all_pose_6d[ind_img, None, :, :].clone().detach().requires_grad_(True)
        optimed_orient_6d = all_orient_6d[ind_img, None, :, :].clone().detach().requires_grad_(True)
        optimed_betas = res["betas"][ind_img, None, :].clone().detach().requires_grad_(True)
        optimed_trans_xy = res["trans"][ind_img, None, :2].clone().detach().requires_grad_(True)
        optimed_trans_z = res["trans"][ind_img, None, 2:3].clone().detach().requires_grad_(True)
        optimed_camera_flength = res["flength"][ind_img, None, :].clone().detach().requires_grad_(True)
        n_vert_comp = 2 * self.smal.n_center + 3 * self.smal.n_left
        optimed_vert_off_compact = torch.zeros(
            (1, n_vert_comp), dtype=torch.float, device=self.device, requires_grad=True
        )
        optimed_betas_limbs = res["betas_limbs"][ind_img, None, :].clone().detach().requires_grad_(True)

        faces_prep = self.smal.faces.unsqueeze(0)

        if not self.apply_ttopt:
            with torch.no_grad():
                optimed_pose_with_glob = get_optimed_pose_with_glob(optimed_orient_6d, optimed_pose_6d)
                optimed_trans = torch.cat((optimed_trans_xy, optimed_trans_z), dim=1)
                smal_verts, keyp_3d, _ = self.smal(
                    beta=optimed_betas, betas_limbs=optimed_betas_limbs,
                    pose=optimed_pose_with_glob, vert_off_compact=optimed_vert_off_compact,
                    trans=optimed_trans, keyp_conf="olive", get_skin=True,
                )
        else:
            smal_verts = self._run_ttopt(
                optimed_pose_6d, optimed_orient_6d, optimed_betas,
                optimed_betas_limbs, optimed_trans_xy, optimed_trans_z,
                optimed_camera_flength, optimed_vert_off_compact,
                faces_prep, res, keypoint_weights, target_dict,
            )

        vertices_np = smal_verts[0].detach().cpu().numpy()
        faces_np = faces_prep[0].detach().cpu().numpy()

        return {
            "vertices": vertices_np,
            "faces": faces_np,
            "betas": optimed_betas[0].detach().cpu().numpy(),
            "betas_limbs": optimed_betas_limbs[0].detach().cpu().numpy(),
            "image_path": image_path,
        }

    def _run_ttopt(self, optimed_pose_6d, optimed_orient_6d, optimed_betas,
                   optimed_betas_limbs, optimed_trans_xy, optimed_trans_z,
                   optimed_camera_flength, optimed_vert_off_compact,
                   faces_prep, res, keypoint_weights, target_dict):
        """Test-time optimization (301 SGD steps)."""
        from src.smal_pytorch.utils import get_optimed_pose_with_glob
        from src.combined_model.loss.loss_utils import reset_loss_values
        from src.combined_model.loss.ttopt_loss_utils import (
            leg_sideway_error, leg_torsion_error,
            tail_sideway_error, tail_torsion_error,
            spine_sideway_error, spine_torsion_error,
            calculate_plane_errors_batch,
        )
        try:
            from pytorch3d.structures import Meshes
            from pytorch3d.loss import mesh_edge_loss, mesh_normal_consistency, mesh_laplacian_smoothing
            HAS_PYTORCH3D = True
        except ImportError:
            HAS_PYTORCH3D = False
            print("[WARN] pytorch3d 없음 - mesh regularization loss 비활성화")

        ind_img = 0
        losses = reset_loss_values(self.loss_weights)

        with torch.no_grad():
            thr_kp = 0.2
            kp_weights = res["hg_keyp_scores"]
            kp_weights[res["hg_keyp_scores"] < thr_kp] = 0
            weights_resh = kp_weights[ind_img, None, :, :].reshape((-1))
            keyp_w_resh = keypoint_weights.repeat((1, 1)).reshape((-1))

            sm = torch.nn.Softmax(dim=1)
            target_gc_class = sm(res["vertexwise_ground_contact"][ind_img, :, :])[None, :, 1]
            target_gc_class_remeshed = torch.einsum(
                "ij,aij->ai", self.remeshing_barys,
                target_gc_class[:, self.remeshing_faces].float()
            )
            target_gc_class_remeshed_prep = torch.round(target_gc_class_remeshed).to(torch.long)

            target_hg_silh = res["hg_silh_prep"][ind_img, :, :].detach()
            target_kp_resh = res["hg_keyp_256"][ind_img, None, :, :].reshape((-1, 2)).detach()

            isflat = [res["isflat_prep"][ind_img] >= 0.5]
            istouching = [target_gc_class_remeshed_prep.sum() > 3]

        optimizer = torch.optim.SGD(
            [optimed_camera_flength, optimed_trans_z, optimed_trans_xy,
             optimed_pose_6d, optimed_orient_6d, optimed_betas, optimed_betas_limbs],
            lr=5e-4, momentum=0.9,
        )
        optimizer_vshift = torch.optim.SGD(
            [optimed_camera_flength, optimed_trans_z, optimed_trans_xy,
             optimed_pose_6d, optimed_orient_6d, optimed_betas,
             optimed_betas_limbs, optimed_vert_off_compact],
            lr=1e-4, momentum=0.9,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, min_lr=1e-5, patience=5,
        )
        scheduler_vshift = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_vshift, mode="min", factor=0.5, min_lr=1e-5, patience=5,
        )

        current_optimizer = optimizer
        current_scheduler = scheduler
        current_weight_name = "weight"
        arap_loss_fn = None
        laplacian_ctf_fn = None

        for i in range(self.ttopt_steps):
            if i == 150:
                current_optimizer = optimizer_vshift
                current_scheduler = scheduler_vshift
                current_weight_name = "weight_vshift"
                if HAS_PYTORCH3D and losses.get("arap", {}).get("weight_vshift", 0) > 0:
                    with torch.no_grad():
                        from src.combined_model.loss.arap import Arap_Loss
                        torch_mesh_cmp = Meshes(smal_verts.detach(), faces_prep.detach())
                        arap_loss_fn = Arap_Loss(meshes=torch_mesh_cmp, device=self.device)

            current_optimizer.zero_grad()

            optimed_pose_with_glob = get_optimed_pose_with_glob(optimed_orient_6d, optimed_pose_6d)
            optimed_trans = torch.cat((optimed_trans_xy, optimed_trans_z), dim=1)
            smal_verts, keyp_3d, _ = self.smal(
                beta=optimed_betas, betas_limbs=optimed_betas_limbs,
                pose=optimed_pose_with_glob, vert_off_compact=optimed_vert_off_compact,
                trans=optimed_trans, keyp_conf="olive", get_skin=True,
            )

            pred_silh_images, pred_keyp_raw = self.silh_renderer(
                vertices=smal_verts, points=keyp_3d,
                faces=faces_prep, focal_lengths=optimed_camera_flength,
            )
            pred_keyp = pred_keyp_raw[:, :24, :]

            diff_silh = torch.abs(pred_silh_images[0, 0, :, :] - target_hg_silh)
            losses["silhouette"]["value"] = diff_silh.mean()

            output_kp_resh = pred_keyp[0, :, :].reshape((-1, 2))
            losses["keyp"]["value"] = (
                (((output_kp_resh - target_kp_resh)[weights_resh > 0] ** 2)
                 .sum(axis=1).sqrt() * weights_resh[weights_resh > 0])
                * keyp_w_resh[weights_resh > 0]
            ).sum() / max((weights_resh[weights_resh > 0] * keyp_w_resh[weights_resh > 0]).sum(), 1e-5)

            losses["pose_legs_side"]["value"] = leg_sideway_error(optimed_pose_with_glob)
            losses["pose_legs_tors"]["value"] = leg_torsion_error(optimed_pose_with_glob)
            losses["pose_tail_side"]["value"] = tail_sideway_error(optimed_pose_with_glob)
            losses["pose_tail_tors"]["value"] = tail_torsion_error(optimed_pose_with_glob)
            losses["pose_spine_side"]["value"] = spine_sideway_error(optimed_pose_with_glob)
            losses["pose_spine_tors"]["value"] = spine_torsion_error(optimed_pose_with_glob)

            sel_verts = torch.index_select(
                smal_verts, dim=1, index=self.remeshing_faces.reshape((-1))
            ).reshape((1, self.remeshing_faces.shape[0], 3, 3))
            verts_remeshed = torch.einsum("ij,aijk->aik", self.remeshing_barys, sel_verts)
            gc_plane, gc_below = calculate_plane_errors_batch(
                verts_remeshed, target_gc_class_remeshed_prep, isflat, istouching,
            )
            losses["gc_plane"]["value"] = torch.mean(gc_plane)
            losses["gc_belowplane"]["value"] = torch.mean(gc_below)

            if HAS_PYTORCH3D:
                mesh_loss_weight = sum(
                    losses.get(k, {}).get(current_weight_name, 0)
                    for k in ["edge", "normal", "laplacian"]
                )
                if mesh_loss_weight > 0:
                    torch_mesh = Meshes(smal_verts, faces_prep.detach())
                    losses["edge"]["value"] = mesh_edge_loss(torch_mesh)
                    losses["normal"]["value"] = mesh_normal_consistency(torch_mesh)
                    losses["laplacian"]["value"] = mesh_laplacian_smoothing(torch_mesh, method="uniform")

                if arap_loss_fn is not None and losses.get("arap", {}).get(current_weight_name, 0) > 0:
                    torch_mesh = Meshes(smal_verts, faces_prep.detach())
                    losses["arap"]["value"] = arap_loss_fn(torch_mesh)

            total_loss = sum(
                losses[k]["value"] * losses[k].get(current_weight_name, 0)
                for k in losses
                if isinstance(losses[k], dict)
                and "value" in losses[k]
                and losses[k].get(current_weight_name, 0) > 0
            )

            total_loss.backward(retain_graph=True)
            current_optimizer.step()
            current_scheduler.step(total_loss)

            if i % 100 == 0:
                print(f"  [ttopt] step {i}/{self.ttopt_steps-1} loss={total_loss.item():.4f}")

        return smal_verts


def main():
    """CLI로 단일 이미지 추론."""
    import argparse

    parser = argparse.ArgumentParser(description="BITE 추론 실행")
    parser.add_argument("--image", required=True, help="강아지 이미지 경로")
    parser.add_argument("--bite-dir", default="bite_release", help="BITE 코드 디렉토리")
    parser.add_argument("--output", default="outputs", help="결과 저장 디렉토리")
    parser.add_argument("--no-ttopt", action="store_true", help="test-time optimization 비활성화")
    parser.add_argument("--device", default="auto", help="cuda/cpu/auto")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    runner = BITERunner(
        bite_dir=args.bite_dir,
        device=args.device,
        apply_ttopt=not args.no_ttopt,
    )

    result = runner.run(args.image)

    name = Path(args.image).stem
    out_path = Path(args.output) / f"{name}_result.pkl"

    import pickle
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\n[완료] 결과 저장: {out_path}")
    print(f"  vertices: {result['vertices'].shape}")
    print(f"  betas: {result['betas'].shape}")

    mesh = trimesh.Trimesh(
        vertices=result["vertices"], faces=result["faces"],
        process=False, maintain_order=True,
    )
    mesh_path = Path(args.output) / f"{name}.obj"
    mesh.export(str(mesh_path))
    print(f"  mesh: {mesh_path}")


if __name__ == "__main__":
    main()
