# Compound Protein Interaction with Graph Matching Network

This project applies Graph Matching Networks (GMN) to predict the binding affinity between proteins and small molecules (ligands), a key task in drug discovery. 

### The Problem
In most standard Graph Neural Network (GNN) approaches, the protein and ligand graphs are processed independently. This independent processing has a fundamental limitation: it prevents the two molecular graphs from exchanging information during the representation learning phase, losing crucial details about how the two structures interact at the node level.

### The Proposed Approach
To overcome this limitation, the project implements a GMN architecture that introduces a bidirectional cross-graph attention mechanism. 
* During the propagation layers (message passing), protein and ligand nodes communicate directly to exchange information.
* This allows the model to dynamically capture structural dependencies and differences between the two molecules, producing a matching vector that is incorporated into the node update step alongside standard intra-graph messages.
* After multiple rounds of this joint propagation, the representations of both graphs are pooled into fixed-size vectors and combined to predict the continuous binding affinity value.

### Key Results & Benefits
* **Better Performance:** Preliminary results on the PDBbind dataset show that the GMN reduces the Mean Squared Error (MSE) and variance across folds compared to a baseline GNN that processes the graphs in isolation. This suggests that cross-graph communication improves prediction accuracy.
* **Interpretability:** The cross-graph attention mechanism offers a natural form of interpretability. The attention weights spontaneously concentrate on biologically meaningful regions of the protein, such as the binding pocket, even without explicit pocket information.
