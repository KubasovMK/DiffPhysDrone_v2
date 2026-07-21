#!/usr/bin/env python3

from collections import OrderedDict
from pathlib import Path

import torch


INPUT_PATH = Path("base_converted.pth")
OUTPUT_PATH = Path("base_converted_back.pth")

RENAME_MAP = {
    "v_proj.weight": "observation_fc.weight",
    "v_proj.bias": "observation_fc.bias",
    "fc.weight": "action_fc.weight",
}


def rename_keys(state_dict):
    converted = OrderedDict()

    for old_key, value in state_dict.items():
        # Поддержка checkpoint, сохранённого через DataParallel
        if old_key.startswith("module."):
            prefix = "module."
            clean_key = old_key[len(prefix):]
        else:
            prefix = ""
            clean_key = old_key

        new_clean_key = RENAME_MAP.get(clean_key, clean_key)
        new_key = prefix + new_clean_key

        if new_key in converted:
            raise RuntimeError(
                f"После переименования возник дублирующийся ключ: {new_key}"
            )

        converted[new_key] = value.clone()

        if old_key != new_key:
            print(
                f"{old_key:30s} -> {new_key:30s} "
                f"shape={tuple(value.shape)}"
            )

    return converted


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Не найден файл: {INPUT_PATH.resolve()}"
        )

    checkpoint = torch.load(INPUT_PATH, map_location="cpu")

    # Обычный state_dict / OrderedDict
    if (
        isinstance(checkpoint, dict)
        and checkpoint
        and all(torch.is_tensor(value) for value in checkpoint.values())
    ):
        output_checkpoint = rename_keys(checkpoint)

    # Полный checkpoint со state_dict
    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
        and isinstance(checkpoint["state_dict"], dict)
    ):
        output_checkpoint = checkpoint.copy()
        output_checkpoint["state_dict"] = rename_keys(
            checkpoint["state_dict"]
        )

    # Полный checkpoint с model_state_dict
    elif (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
        and isinstance(checkpoint["model_state_dict"], dict)
    ):
        output_checkpoint = checkpoint.copy()
        output_checkpoint["model_state_dict"] = rename_keys(
            checkpoint["model_state_dict"]
        )

    else:
        raise RuntimeError(
            "Не удалось определить структуру checkpoint. "
            f"Тип объекта: {type(checkpoint)}"
        )

    torch.save(output_checkpoint, OUTPUT_PATH)

    print()
    print("Готово.")
    print(f"Исходный файл: {INPUT_PATH.resolve()}")
    print(f"Новый файл:    {OUTPUT_PATH.resolve()}")
    print("Значения весов не изменялись — изменены только имена ключей.")


if __name__ == "__main__":
    main()