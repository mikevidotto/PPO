# TODO

- [ ] optimize the values network
    - [ ] get the returns for each step
    - [ ] get the MSE loss for each step, then average it.
    - [ ] implement backpropagation to update the weights within the values network
        - [ ] find the loss w.r.t weights
        - [ ] ask about loss w.r.t. biases
        - [ ] average the gradients
        - [ ] apply them to current values network using the learning rate
