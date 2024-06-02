#  correlation matrix heatmap
corrmat = white.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sn.heatmap(corrmat, annot=True, cmap="PuBu")
plt.show()
