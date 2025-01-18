import onnxruntime as ort
import openvino_genai
import openvino
import torch.utils.data as Data
import dataset
import data_prepare
import torch

def get_pad_mask(seq:torch.Tensor, vocab_pad_value=0):
    # seq: [batch_size, seq_len]
    # pad_mask: [batch_size, 1, 1, seq_len]
    batch_size, seq_len = seq.shape
    # pad_mask = seq.data.eq(vocab_pad_value)
    pad_mask = torch.eq(seq, float(vocab_pad_value))
    pad_mask = pad_mask.unsqueeze(1).unsqueeze(1)
    # pad_mask = pad_mask.expand(batch_size, 1, seq_len, seq_len)
    # print("pad_mask shape", pad_mask.shape)
    return pad_mask

def get_subseq_mask(seq:torch.Tensor):
    # seq: [batch_size, seq_len]
    # subseq_mask: [batch_size, 1, seq_len, seq_len]
    batch_size, seq_len = seq.shape
    subseq_mask = torch.triu(torch.ones([batch_size, 1, seq_len, seq_len]), diagonal=1).bool()
    # print("subseq_mask shape", subseq_mask.shape)
    return subseq_mask

def greedy_decode(session, enc_input, start_symbol, end_symbol, max_len):
    # 1. Encode
    enc_mask = get_pad_mask(seq=enc_input, vocab_pad_value=0).to(device=dev)
    enc_outputs = model.Encoder(enc_input, mask=enc_mask)

    # 2. Init decoder input
    dec_input = torch.zeros(1, max_len).type_as(enc_input.data).to(device=dev)
    next_symbol = start_symbol

    # 3. greedy decoding
    for i in range(max_len):
        dec_input[0][i] = next_symbol
        dec_pad_mask = get_pad_mask(seq=dec_input, vocab_pad_value=0).to(device=dev)
        dec_subseq_mask = get_subseq_mask(seq=dec_input).to(device=dev)
        dec_mask = dec_pad_mask | dec_subseq_mask
        # print("===========")
        # print("dec_input shape", dec_input.shape, "enc_output shape", enc_outputs.shape)
        # print("===========")
        dec_outputs = model.Decoder(dec_input, enc_outputs, mask=dec_mask)
        projected = model.projection(dec_outputs)
        next_symbol = projected.squeeze(0).max(dim=-1)[1][i].item()

        if next_symbol == end_symbol:
            break
    
    # 4. collect predictions
    return dec_input
def inference(session, loader:Data.DataLoader):
    src_vocab, tgt_vocab = data_prepare.get_vocab()
    enc_inputs, _, _ = next(iter(loader))
    # print("enc_inputs ", enc_inputs.shape)
    enc_inputs = enc_inputs.to(dev)

    start_symbol = tgt_vocab['S']
    end_symbol = tgt_vocab['E']
    max_len = 64

    # Get the decoder input using greedy decoding
    # print("greedy input", enc_inputs)
    # print("start idx", start_symbol, "end idx", end_symbol)
    predict_dec_input = greedy_decode(model, enc_inputs, start_symbol, end_symbol, max_len)

    # Run the model to get the final predictions
    predict = model(enc_inputs, predict_dec_input)

    # Extract the most probable predictions
    final_predictions = predict.data.max(-1)[1]

    print("Input:", enc_inputs)
    print([loader.dataset.src_idx2word[int(i)] for i in enc_inputs[0]])
    print("Final Predictions:", final_predictions)
    print([loader.dataset.tgt_idx2word[int(i)] for i in final_predictions])

print(openvino_genai.__path__)
# print(openvino_genai.openvino.)
# set "PYTHONPATH=C:\\Users\\Chenghong\\miniconda3\\envs\\torch-xpu\\Lib\\site-packages\\openvino_genai"
# set "PATH=%PATH%;C:\\Users\\Chenghong\\miniconda3\\envs\\torch-xpu\\Lib\\site-packages\\openvino_genai\bin"

print(openvino.Core().available_devices)
device = "GPU"

print(openvino.Core().get_property(device, "FULL_DEVICE_NAME"))

core = openvino.Core()

print(ort.__version__)
print(ort.get_available_providers() )
# Set the OpenVINO™ Execution Provider
providers = [('OpenVINOExecutionProvider', {'device_type': 'GPU'})]
# Load the ONNX model

# model = core.read_model("transformer_fp16.onnx")
# compiled_model = core.compile_model(model, device)

session = ort.InferenceSession("transformer_fp16.onnx", providers=providers)
# session = ort.InferenceSession(compiled_model, providers=providers)
# Prepare input data 
# input_name = session.get_inputs()[0].name
input_name = session.get_inputs()
print(input_name[0].name, input_name[1].name)
# exit()


test_dataset = dataset.MyDataSet(filename='test_data.txt')
test_loader = Data.DataLoader(test_dataset, batch_size=1, shuffle=True)
for i in range(100):
    enc_input, dec_input, _ = next(iter(test_loader))
    # Run inference
    result = session.run(None, {'enc_inputs': enc_input.numpy(), 'dec_inputs': dec_input.numpy()})
    print(result)