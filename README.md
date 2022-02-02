# Depression-Engine
Depression Detection Based on Speech activity
in this project we are trying to observe and study depression effect on speech
mostly on speech activity, meaning number of silences in speech and length of this silences 
first we need to exctract silences from speech, we are going to do this wtith our SAD model
then after extracting silences, we perform two-tailed t-test
then we are going to use Spectrograms exctracted from speech to detect depression

# DATASET

This database is part of a larger corpus, the Distress Analysis Interview Corpus (DAIC) (Gratch et al.,2014), that contains clinical interviews designed to support the diagnosis of psychological distress conditions such as anxiety, depression, and post-traumatic stress disorder. These interviews were collected as part of a larger effort to create a computer agent that interviews people and identifies verbal and nonverbal indicators of mental illness (DeVault et al., 2014). Data collected include audio and video recordings and extensive questionnaire responses; this part of the corpus includes the Wizard-of-Oz interviews, conducted by an animated virtual interviewer called Ellie, controlled by a human interviewer in another room. Data has been transcribed and annotated for a variety of verbal and non-verbal features.

This share includes 189 sessions of interactions ranging between 7-33min (with an average of 16min). Each session includes transcript of the interaction, participant audio files, and facial features. For more details please refer to the [documentation](https://dcapswoz.ict.usc.edu/wwwutil_files/DAICWOZDepression_Documentation.pdf)

you can download this dataset at [Download](https://dcapswoz.ict.usc.edu/)


# SAD(Speech Activity Detection)

first we use VUV(voiced/unvoiced) provided by the dataset itself for our target label, which is not a tottaly accurate assumption

we exctracted MFFCs and their deltas from interviews, and use them as input for our model

model is a hybrid model using both CNN and GRUs
BatchNormalization after CNNs for training speedup
using Dropout after each layer of GRUs 
and the befor the output layer we used a GlobalAveragePooling Layer

we used tf.Keras for defining model and google colab for Training, and we achived the result in blew table

| _ | Train_set | Validation_set | Test_set |
| ------------- | ------------- | ------------- |------------- |
|AUC : | 0.9457  | 0.9432  |0.9290  |




# Depression Detection 

first we cleaned interviews audio files, to contain only Participants Voice
then used [Audiomentations](https://github.com/iver56/audiomentations) a publicly available python library, to augment Train_set
then splited each audio file to 2-Secs samples and exctracted Spectrogram for each of them
we used a Conv_1d Model for this purpose and managed to achive 0.86 recall on the test set, which means we could mostly identify depressed people


