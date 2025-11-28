import numpy as np

# Load predictions
data = np.load('results/vgg_baseline_predictions.npz')
preds = data['predictions']
gt = data['ground_truth']

print('VGG Predictions distribution:')
print(f'  Choice 1: {np.sum(preds==1):6d} ({np.mean(preds==1)*100:5.2f}%)')
print(f'  Choice 2: {np.sum(preds==2):6d} ({np.mean(preds==2)*100:5.2f}%)')
print(f'  Choice 3: {np.sum(preds==3):6d} ({np.mean(preds==3)*100:5.2f}%)')
print(f'  Total:    {len(preds):6d}')

print('\nGround truth distribution:')
print(f'  Choice 1: {np.sum(gt==1):6d} ({np.mean(gt==1)*100:5.2f}%)')
print(f'  Choice 2: {np.sum(gt==2):6d} ({np.mean(gt==2)*100:5.2f}%)')
print(f'  Choice 3: {np.sum(gt==3):6d} ({np.mean(gt==3)*100:5.2f}%)')
print(f'  Total:    {len(gt):6d}')

print(f'\nAccuracy: {np.mean(preds == gt)*100:.2f}%')
print(f'\nConclusion:')
if np.max([np.mean(preds==1), np.mean(preds==2), np.mean(preds==3)]) > 0.9:
    print('⚠️ WARNING: VGG is mostly predicting one label (degenerate)')
else:
    print('✓ VGG predictions are balanced - not degenerate')
