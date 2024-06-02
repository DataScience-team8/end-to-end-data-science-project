#  correlation matrix heatmap
corrmat = white.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sn.heatmap(corrmat, annot=True, cmap="PuBu")
plt.show()
![image](https://github.com/DataScience-team8/end-to-end-data-science-project/assets/126345795/a062d779-bd27-42cd-8cd6-162b02c7a5c8)
