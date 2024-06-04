#!/bin/bash

dictionary_path=/home/hoang/PycharmProjects/naturalspeech/phoneme/phone_dict/viIPA.txt
output_model_path=/home/hoang/PycharmProjects/naturalspeech/phoneme/g2p_model/viIPA_model.zip

mfa train_g2p $dictionary_path $output_model_path
