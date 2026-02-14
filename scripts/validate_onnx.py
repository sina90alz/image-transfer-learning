from pathlib import Path
import onnx

def main():
    p = Path("artifacts/model.onnx")
    m = onnx.load(str(p))
    onnx.checker.check_model(m)
    print("✅ ONNX model is structurally valid.")
    print("Opset imports:", [(o.domain, o.version) for o in m.opset_import])
    print("Inputs:", [i.name for i in m.graph.input])
    print("Outputs:", [o.name for o in m.graph.output])

if __name__ == "__main__":
    main()
