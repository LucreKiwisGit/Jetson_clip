import onnx

def print_model_info(model_path):
    model = onnx.load(model_path)
    print("Model Ir Version:", model.ir_version)
    print("Model Opset Import:", model.opset_import)
    print("Model Producer Name:", model.producer_name)
    print("Model Graph Name:", model.graph.name)
    print("Model Inputs:")
    for input in model.graph.input:
        print(f"  - {input.name}: {input.type.tensor_type.elem_type}, Shape: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")

    print("Model Outputs:")
    for output in model.graph.output:
        print(f"  - {output.name}: {output.type.tensor_type.elem_type}, Shape: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")

    print("Model Nodes:")
    for node in model.graph.node:
        print(f"  - {node.op_type}: {node.name}, Inputs: {node.input}, Outputs: {node.output}")

if __name__ == "__main__":
    model_path = "./data/model/resnet/checkpoint.onnx"  # 替换为你的模型路径
    print_model_info(model_path)
