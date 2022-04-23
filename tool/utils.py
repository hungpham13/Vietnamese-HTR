def batch_to_device(batch, device):
    img = batch['img'].to(device, non_blocking=True)
    tgt_input = batch['tgt_input'].to(device, non_blocking=True)
    tgt_output = batch['tgt_output'].to(device, non_blocking=True)
    tgt_padding_mask = batch['tgt_padding_mask'].to(device, non_blocking=True)

    return {
            'img': img, 'tgt_input':tgt_input,
            'tgt_output':tgt_output, 'tgt_padding_mask':tgt_padding_mask,
            'filenames': batch['filenames']
            }