# Week 5 Summary (11/27 - 12/3)

## Project: Project 7b - Neural Representation of Massive 3D Data
- Supervisor: Jens Sjölund <jens.sjolund@it.uu.se>
- Group Members:
    - Aoping Lyu <aoping.lyu.6830@student.uu.se>
    - Jinglin Gao <jinglin.gao.1941@student.uu.se>

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