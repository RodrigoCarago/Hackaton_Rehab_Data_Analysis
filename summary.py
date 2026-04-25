'''
BCI data from patients recovering from stroke

The EEG dataset has data recorded on the first "pre" and lost "post" sessions of the rehabilitation process with each session having
a test and train run.

In each trial the user should immagine either a left- or right-hand movement. At t=0, the EEG data is marked with a 1 or -1 (for left or 
right hand, respectively). At t=2 the user receives the command and should initiate the movement thought. Training data is used for 
training at every session. During training the user receives feedback using FES and visual cues. Then, on the test run, the model is 
tested when the user receives directions about what to do with feedback only being provided if the  model can correctly predict the 
user's intention. Allows to get a performance metric.


'''