from inference import SuperResolutionModel

# Load model once
sr_model = SuperResolutionModel('checkpoints/best_weights_export/model_weights.pth')

# Use multiple times
sr_image1 = sr_model.upscale('image1.png')
sr_image2 = sr_model.upscale('image2.png', 'output2.png')

# Batch process
sr_model.upscale_batch('input_folder/', 'output_folder/')