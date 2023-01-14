def birds(): 
    import os
    import csv
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    #load dataset into Pandas DataFrame
    originaldf=pd.read_csv('birds.csv')

    # STEP 1: Standardization (standardize the dataset’s features onto unit scale (mean = 0 and variance = 1)
    scaler=StandardScaler()
    standardised_data = scaler.fit_transform(originaldf)
    df_standardised_data = pd.DataFrame(standardised_data, columns=['BodyMass', 'Wingspan'])

    # STEP 2: Run PCA into 1 dimension
    pca = PCA(n_components=1)
    pca_data = pca.fit_transform(standardised_data)
    pcaDf = pd.DataFrame(data = pca_data, columns = ['principal component 1'])

    # STEP 3: Reconstruct the original data 
    reconstructed_standardised_data = pca.inverse_transform(pca.transform(standardised_data))
    df_reconstructed_standardised_data = pd.DataFrame(reconstructed_standardised_data, columns=["BodyMass","Wingspan"])
    reconstructed_original_data = scaler.inverse_transform(reconstructed_standardised_data)
    df_reconstructed_original_data = pd.DataFrame(reconstructed_original_data, columns=["BodyMass","Wingspan"])

    # STEP 4: Compute the reconstruction error using MSE
    mse = np.sum((originaldf- reconstructed_original_data)**2, axis=1).mean()
    print("Reconstruction error:", mse)


    #Plot data
    print("Original Data:")
    originaldf.plot(kind="scatter",x="BodyMass",y="Wingspan")
    plt.show()
    print("Preprocessed Data:")
    df_standardised_data.plot(kind="scatter",x="BodyMass",y="Wingspan")
    plt.show()
    print("Data projected into 1D using PCA")
    pcaDf.plot(kind= "scatter",x='principal component 1',y=0)
    plt.show()
    print("reconstructed standardised data")
    df_reconstructed_standardised_data.plot(kind="scatter",x="BodyMass",y="Wingspan", 
    title = 'reconstructed standardised data',color = "blue")
    plt.show()
    print("Reconstructed Data:")
    df_reconstructed_original_data.plot(kind="scatter",x="BodyMass",y="Wingspan")
    plt.show()
    print("Reconstructed data + post-processing (mean, std):")
    plt.scatter(reconstructed_original_data[:,0], reconstructed_original_data[:,1])
    plt.title("post-processed reconstructed data (mean, sd)")
    mean_x = reconstructed_original_data.mean()
    mean_y = reconstructed_original_data.mean()
    std = reconstructed_original_data.std()
    plt.errorbar(mean_x, mean_y, std, marker="o", color = 'green',  markerfacecolor="red")
    plt.show()
    print("mean BodyMass")
    print(np.mean(reconstructed_original_data[0])) #mean BodyMass
    print("mean Wingspan")
    print(np.mean(reconstructed_original_data[1])) #mean Wingspan
    print("std BodyMass")
    print(np.std(reconstructed_original_data[0]))  #std BodyMass
    print("std Wingspan")
    print(np.std(reconstructed_original_data[1]))  #std Wingspan

def component_selection(x):
    import os
    import csv
    import numpy as np
    import pandas as pd
    import sklearn
    from matplotlib import pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    df= sklearn.datasets.fetch_california_housing(as_frame=True)
    df= pd.DataFrame(data=df.data,columns=df.feature_names)
    # Standardizing the features
    scaler=StandardScaler()
    df1= scaler.fit_transform(df)
    #pca
    pca = PCA(n_components=x)
    principalComponents = pca.fit_transform(df1)

    #reconstruct original data
    reconstructedarray=pca.inverse_transform(pca.transform(df1))
    reconstructedoriginal=scaler.inverse_transform(reconstructedarray)

    #compute reconstruction error using MSE
    mse= np.sum((df - reconstructedoriginal)**2, axis=1).mean()
    print("Reconstruction Error:",mse)
    return mse
