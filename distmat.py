import matplotlib.pyplot as plt

# Create figure
fig, ax = plt.subplots(figsize=(8, 4))

# Add arrows
ax.arrow(0.2, 0.5, 0.35, 0, head_width=0.08, color='k') 
ax.arrow(0.55, 0.5, -0.35, 0, head_width=0.08, color='k')
ax.arrow(0.2, 0.35, 0.35, 0, head_width=0.08, color='k')  
ax.arrow(0.55, 0.35, -0.35, 0, head_width=0.08, color='k')

# Add text  
ax.text(0.05, 0.5, 'C1', fontsize=14)
ax.text(0.8, 0.5, 'C2', fontsize=14)
ax.text(0.275, 0.22, 'Project C1 onto E2', fontsize=10) 
ax.text(0.275, 0.38, 'Project C2 onto E1', fontsize=10)
ax.set_title('Distance Matrix Calculation', fontsize=16)

# Save figure
plt.savefig('vqpca_distmat.png', bbox_inches='tight')