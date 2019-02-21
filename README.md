# Dense-Captioning
Deep Learning Model which generates Captions/Description of a given video
## Problem Statement
Dense Captioning is a system inspired by the paperAutomatic generation of video descriptions using state of the art encoder-decoder deep learning models.

## DataSet
Microsoft Research - Video to Text (MSR-VTT) is a dataset composed of 10k Videos collected from YouTube and annotated using a crowdsourcing platform. Each video has a variable number of captions which totals upto 200000 captions, annotated by different users. We used 80% videos as train and 20% video as test split. This DataSet contains approx. 1600 cartoon videos which includes kid cartoons, video games, and anime videos.

## Methodology
Following figure depicts our proposal for caption generation on MSR-VTT dataset. We used the popular encoder-decoder architecture utilizing both state of the art CNNs and LSTMs along with attention mechanism for sequence to sequence translation.

## The encoding Stage

Feature Extraction
First(Blue in the figure) We fed raw video frames to the state of the art CNN (VGG16) to extract complementary features. As processing complete frames was not computationally feasible so we used equally spaced 100 frames from each video.  We removed last classification layers to capture more general features. Furthermore, for the sake of reducing computational load we added a linear layer to compress 2048 features to 128 per frame and adjust its weight through an autoEncoder. An autoEncoder tries to regenerate its input while minimizing reconstruction loss. If we decrease the size of hidden layer it learns to represent 2048 features into 128 by compressing it to only its important latent features.


## Context vector encoding
Second(Red In the scheme) As We needed to describe the actions performed in the consecutive frames We considered Bidirectional LSTM (Long Short Term memory model) for this purpose. LSTMs generally work better than RNN and GRU units as it utilizes an extra hidden state to save temporal dependencies between the frames and avoids vanishing gradient problem. Moreover, Uni-directional (Forward Only) LSTM generates next output by only looking at its previous inputs. However, in natural languages  ‘Context’ plays an important role and helps to translate precisely. For this purpose we used a BLSTM (a bidirectional LSTM) which utilizes the forward and backward pass in time to be more context aware. The hidden states from both LSTMs (forward and backward) is passed through a linear layer that compresses them into a single context vector.  

## Decoding stage

3rd (Yellow in scheme) this context vector is then fed into a the soft attention model along with the encoder outputs. Without the attention mechanism the whole information is encoded into the hidden states. However, when sequences become longer, only hidden states cannot bare the burden of complete sentence translation. We used an attention mechanism which utilizes hidden states as well as previous encoder outputs. This attention model decides on which parts of the input video it should focus for emitting the next word, considering the description generated so far. 


Fourth(green in the scheme) Another BLSTM which takes previous hidden states , encoder outputs and previous predicted word to generate next word. We have used an embedding layer instead of one-hot vector for word representation. The complete variable-length caption is generated word by word using a log softmax at the top of this BLSTM.


# Evaluation Metrics
Inspired by various prior works in image and video captioning we evaluate our model on traditional evaluation metrics: 


## Bleu
This algorithm is used to evaluate the quality of text generated for machine translation. Here the quality is determined by how much the machine generated text is similar to the human ground truth


## METEOR
This algorithm scores machine translation hypotheses by aligning them to one or more reference (usually human labelled) translations. Alignments are based on exact, stem, synonym, and paraphrase matches between words and phrases.

## CIDEr
This is measured by calculating the cosine similarity score for the n-grams in the reference and the generated captions. Basically the aim to calculate how many n-grams are common between the generated and reference caption and how many are exclusive to each.


## SPICE
First the candidate and reference caption is converted into a semantic graph that ignores stop words and puts emphasis on relations between the objects and their qualities in the caption. Then the graphs are compared to measure their similarity.

Albeit Bleu and Meteor were created to evaluate accuracy for machine translation or summary generation many prior captioning models use these to evaluate captions. Therefore, to compare our model to the prior ones we would have to use these evaluation metrics as well. SPICE and CIDEr on the other hand was created for the sole purpose of evaluating image captions therefore its results would be more accurate.

We use our trained model to generate captions for the test videos and then evaluate the accuracy of captions by comparing them with the ground truths using the metrics mentioned above.

# Results:

Bleu      0.225</br>
CIDEr     0.234</br>
ROUGE_L   0.234</br>
METEOR    0.076</br>
SPICE     0.065



# Examples:

Some Correct descriptions predicted by our model:


Predicted Sentence:   a scene of a video game



Predicted Sentence: a band is performing a song  



Predicted Sentence: a cartoon character talks in front of a cartoon talks




Predicted Sentence: a person stirs ingredients meat in a pan








Examples where our model could not correctly predict and somehow lost it


Predicted Sentence: two wrestling are men in a player and a ball in a ball the the





Predicted Sentence:  a clip of the man a dummy SOS a road




Predicted Sentence: a cartoon character is knights to his animals

