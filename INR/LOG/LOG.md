# Project: Project 7b - Neural Representation of Massive 3D Data
- Supervisor: Jens Sjölund <jens.sjolund@it.uu.se>
- Group Members:
    - Aoping Lyu <aoping.lyu.6830@student.uu.se>
    - Jinglin Gao <jinglin.gao.1941@student.uu.se>

## LOG Link
- Onedrive link: https://uppsalauniversitet-my.sharepoint.com/:t:/g/personal/aoping_lyu_6830_student_uu_se/Ed1AyWlE5u5EqmeppcErzR4BzdD1Qr-AGIYMpU28HJAfcg?e=b4gnSg



## Week 1 Summary (10/30 - 11/5)



### Tuesday (10/31)
- Meeting with the supervisor.

### Thursday (11/2)
- Meeting with the supervisor and the project course convenor.
- Finished downloading the raw files.

### Friday (11/3)
- Identified two possible related open-source libraries:
    - [SlicerMorph](https://github.com/SlicerMorph/SlicerMorph): An extension for importing microCT data and conducting 3D morphometrics in Slicer.
    - [Draco](https://github.com/google/draco): A library for compressing and decompressing 3D geometric meshes and point clouds.


## Week 2 Summary (11/6 - 11/12)

### Read papers
- Neural Fields in Visual Computing and Beyond (For general concepts gaining)
    - Investigate general concepts of neural fields and methods exists for 3D image reconstruction
    - Neural fields can parameterize density domain by directly predicts the density value at a 3D spatial coordinate, and the process is supervised by mapping its output to the senosr domain via Radon(CT) or Fourier(MRI) transform.
- NeRP: Implicit Neural Representation Learning with Prior Embedding for Sparsely Sampled Image Reconstruction
    - MRI reconstruction (Fourier transform)
    - CT reconstruction (Radon transform)
- CoIL: Coordinate-based Internal Learning for Imaging Inverse Problems
    - CT reconstruction
    - Training MLP on the coordinate-response pairs to build a full measurement neural representation.
    

### Work for next week
- Done some literature review
- Start trying out the representation methods

### Proposed Pipeline:
- Data Pre-procesessing and Representation
- Construct Neural Network Model
- Training the Model
- Encoding and Decoding
- Optimization and Evaluation
- Visualisation and Interaction
- Downstream Application




## Week 3 Summary (11/13 - 11/19)

### Key Paper Review
#### SCI: A Spectrum Concentrated Implicit Neural Compression for Biomedical Data

- Paper link: https://doi.org/10.48550/arXiv.2209.15180
    - Focuses on compressing biomedical data, distinct from natural images/videos.
    - Introduces Spectrum Concentrated Implicit neural compression (SCI) based on Implicit Neural Representation (INR).
    - Adapts data into blocks matching INR’s concentrated spectrum envelope.
    - Utilizes a funnel-shaped network for efficient data representation.
    - Implements a parameter allocation strategy for accurate data representation.

    

### Work for next week
- Reproduce the implicit neural compression method using in the found paper.
- Apply it to the 3D data we have.

### Proposed Pipeline:
- Data Pre-procesessing and Representation
- Construct Neural Network Model
- Training the Model
- Encoding and Decoding
- Optimization and Evaluation
- Visualisation and Interaction
- Downstream Application


## Week 4 Summary (11/20 - 11/26)


### Key Paper Review
#### Implicit Neural Representations with Periodic Activation Functions

- Github Link: https://github.com/TT27Bon/INR-project.git
- Start build the MLP model for the data

### Work for next week
- Work on the MLP

### Proposed Pipeline:
- Data Pre-procesessing and Representation
- Construct Neural Network Model
- Training the Model
- Encoding and Decoding
- Optimization and Evaluation
- Visualisation and Interaction
- Downstream Application



## Week 5 Summary (11/27 - 12/3)


### Key Paper Review
#### Implicit Neural Representations with Periodic Activation Functions

- Paper link: https://doi.org/10.48550/arXiv.2209.15180


### Code 

- Github Link: https://github.com/TT27Bon/INR-project.git


### Progress
- Set up the environment for the project on Alvis of C3SE **[AL]** **[JG]**
- Uploaded the raw data to the UPPMAX server **[AL]**
- Tested the example dataset with the open-source code of the paper **[AL]**
- We have converted the raw data into TIFF format for MLP training **[AL]** 
- The core part of the raw data has been extracted and saved as a separate file, the file size is 512*512*512 **[JG]**
- Customized the open source MLP module, which uses SIREN as basic model.  **[JG]**
- Started to study the SIREN **[JG]**


### Meeting with the supervisor - 11/29
- Find some matrices to define the compression rate or the compression ratio to evaluate the compression performance.
- The center of the data is the most important part, so we can use the center part to train the model.
- Look into basic SIREN method and try to implement it on the data.
- Need to start to research on the interface between the MLP and the CNN for further application.

### Work for next week
- Study the SIREN
- Work on the MLP


### Proposed Pipeline:
- Data Pre-procesessing and Representation
- Construct Neural Network Model
- Training the Model
- Encoding and Decoding
- Optimization and Evaluation
- Visualisation and Interaction
- Downstream Application


## Week 6 Summary (12/4 - 12/10)



### Key Paper Review
#### Implicit Neural Representations with Periodic Activation Functions

- Paper link: https://doi.org/10.48550/arXiv.2209.15180


### Code 

- Github Link: https://github.com/TT27Bon/INR-project.git


### Progress
- Finish experiment with Siren **[JG]**
- Write report skeleton **[AL]** **[JG]**
- Use PSNR evaluate the compression result, define compression rate **[JG]**
- Run SCI experiment with cropped data of the tumour centre **[AL]**
- Compare the results between SCI and Siren, choose one as input of the interface **[AL]** **[JG]**

### Oral presentation training - 12/04
- Clear the structure of the presentation   
- Potentially add rhetorical questions to the presentation for better interaction with the audience and smooth transition between different parts of the presentation
- Point to the figures in the presentation to make the presentation more clear
- Provide silence to the audience to let them think about the content
- Apply more time to the pipeline part of the presentation


### Work for next week
- Study the interface between MLP and CNN
- Construct interface


### Proposed Pipeline:
- Data Pre-procesessing and Representation **(DONE)**
- Construct Neural Network Model **(DONE)**
- Training the Model **(DONE)**
- Encoding and Decoding     **(Partially DONE)**
- Optimization and Evaluation
- Visualisation and Interaction
- Downstream Application