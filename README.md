# Moneynet
Fully Unsupervised Learning for Continual Sequence when High Local Correlation

## Todo
### New feature
- [ ] Multi target estimation: baro 
- [ ] Limited monte carlo with target space: baro
- [ ] Shared hidden space algorithm: baro 
### Enhancement
- [ ] Low dimension hidden space available: ljh93
    * cdim 100 hdim 400 - [reject]
    * cdim 20  hdim 400 - [accept], **note**: as decreasing think iteration, temperature softmax needs more selection
    * cdim 20  hdim 400 temperature 0.005 - [reject] 
    * attention for announcing to model with more information
        * probability training with hidden space coefficient is superposition state
        * attention temperature inverse proportion to `int(hdim/indim)` that total iteration
            * other method that concern is attention with learnable parameter
            * if model can't infer well, we should raise temperature  
- [ ] Check relay method effect with low dim hidden space: ljh93
### Lab
#### Recovery
- [x] Hidden space regularization without relay mask: baro
- [x] Hidden space regularization with relay mask replace with hidden space energy loss: baro
- [x] Hidden space regularization without relay mask, hidden space energy loss and residual: baro
- [x] Rollback
- **Conclustion** Loss error make weird loss graph and we success recover that but don't know why
#### Loss with energy
- [ ] Loss with energy 
#### Attention change
- [ ] nan attention case check
- [ ] Select type 1 and type 2 attention and target