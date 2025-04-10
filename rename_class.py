import torch
import sys

# Step 1: Import your new class
from pico_llm import TransformerModelNoPos, Block, MultiHeadAttention, Head, FeedForward, TransformerModelAbsPos, TransformerModelAbsRelPos , TransformerModelRelPos, MultiHeadAttentionRelPos, HeadRelPos, BlockRelPos

# Step 2: Trick Python into mapping old class name to new class
sys.modules['__main__'].TransformerModel = TransformerModelAbsRelPos # 'TransformerModel' = old name
sys.modules['__main__'].Block = BlockRelPos
sys.modules['__main__'].MultiHeadAttention = MultiHeadAttentionRelPos
sys.modules['__main__'].Head = HeadRelPos


# Step 3: Load the model (now mapped correctly)
model = torch.load('temp.pth')

# Step 4: Save it again with the correct class info
torch.save(model, 'rel_plus_abs_pos_transformer.pth')

print("Model re-saved with new class name!")