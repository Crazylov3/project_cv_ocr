import paddle


def load_model(model, params_path):
    ckpt = paddle.load(params_path)

    state_dict = model.state_dict()
    new_state_dict = {}
    for key, value in state_dict.items():
        if key not in ckpt:
            print("{} not in loaded params {} !".format(
                key, ckpt.keys()))
            continue
        pre_value = ckpt[key]
        if list(value.shape) == list(pre_value.shape):
            new_state_dict[key] = pre_value
        else:
            print(
                "The shape of model params {} {} not matched with loaded params shape {} !".
                    format(key, value.shape, pre_value.shape))
    model.set_state_dict(new_state_dict)
