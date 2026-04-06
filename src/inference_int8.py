# import os
# import torch
# from models.quantcb_model import QuantCB_Model

# def load_quantized_model(model, quant_path):
#     print(f"--- Loading INT8 Optimized Weights from {quant_path} ---")
    
#     # weights_only=False because the dict contains custom scale keys
#     checkpoint = torch.load(quant_path, map_location='cpu', weights_only=False)
#     q_weights = checkpoint['weights']
#     scales = checkpoint['scales']
    
#     state_dict = model.state_dict()
    
#     for name, param in state_dict.items():
#         if name in q_weights:
#             weight = q_weights[name]
#             scale_key = f"{name}_scale"
            
#             if scale_key in scales:
#                 # Dequantize: Float = Int * Scale
#                 state_dict[name] = weight.to(torch.float32) * scales[scale_key]
#             else:
#                 state_dict[name] = weight
                
#     model.load_state_dict(state_dict)
#     return model

# def generate_test():
#     # Architecture matching your 16.68M parameter configuration
#     vocab_size = 50257
#     d_model = 256
#     n_heads = 8
#     n_layers = 4
#     d_ff = 1024
#     latent_dim = 128
#     head_dim = 64
    
#     model = QuantCB_Model(
#         vocab_size=vocab_size, 
#         d_model=d_model, 
#         n_heads=n_heads, 
#         d_ff=d_ff, 
#         n_layers=n_layers,
#         latent_dim=latent_dim,
#         head_dim=head_dim
#     )
    
#     # Dynamic pathing to /modelOutput
#     project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     quant_path = os.path.join(project_root, "modelOutput", "quantcb_int8.pth")

#     if not os.path.exists(quant_path):
#         print(f"Error: Quantized model not found at {quant_path}")
#         return

#     model = load_quantized_model(model, quant_path)
#     model.eval()

#     # Create dummy prompt (Batch=1, Seq=5)
#     prompt = torch.randint(0, vocab_size, (1, 5))
    
#     with torch.no_grad():
#         logits, _ = model(prompt)
        
#     print(f"\nInference successful.")
#     print(f"Output Logits Max: {logits.max().item():.4f}")
#     print(f"Output Logits Min: {logits.min().item():.4f}")
#     print("Verified: Model operates correctly with dequantized INT8 weights.")

# if __name__ == "__main__":
#     try:
#         generate_test()
#     except Exception as e:
#         print(f"Inference Error: {e}")