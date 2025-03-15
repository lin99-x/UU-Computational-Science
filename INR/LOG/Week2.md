# Week 2 Summary (11/6 - 11/12)

## Project: Project 7b - Neural Representation of Massive 3D Data
- Supervisor: Jens Sjölund <jens.sjolund@it.uu.se>
- Group Members:
    - Aoping Lyu <aoping.lyu.6830@student.uu.se>
    - Jinglin Gao <jinglin.gao.1941@student.uu.se>

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