---
id: todo
aliases: []
tags: []
---

# TODO

- [ ] optimize the values network
    - [ ] get the returns for each step
    - [ ] get the MSE loss for each step, then average it.
    - [ ] implement backpropagation to update the weights within the values network
        - [ ] find the loss w.r.t weights
        - [ ] ask about loss w.r.t. biases
        - [ ] average the gradients
        - [ ] apply them to current values network using the learning rate

- [ ] modularize methods into their own packages so that there isn't just one huge main.go file.
    - [ ] create a folder structure
    - [ ] put models into each of their own locations
    - [ ] environment
    - [ ] networks
    - [ ] transition data
