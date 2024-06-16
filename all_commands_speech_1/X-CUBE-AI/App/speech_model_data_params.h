/**
  ******************************************************************************
  * @file    speech_model_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    Thu May 16 22:39:05 2024
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef SPEECH_MODEL_DATA_PARAMS_H
#define SPEECH_MODEL_DATA_PARAMS_H
#pragma once

#include "ai_platform.h"

/*
#define AI_SPEECH_MODEL_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_speech_model_data_weights_params[1]))
*/

#define AI_SPEECH_MODEL_DATA_CONFIG               (NULL)


#define AI_SPEECH_MODEL_DATA_ACTIVATIONS_SIZES \
  { 13440, }
#define AI_SPEECH_MODEL_DATA_ACTIVATIONS_SIZE     (13440)
#define AI_SPEECH_MODEL_DATA_ACTIVATIONS_COUNT    (1)
#define AI_SPEECH_MODEL_DATA_ACTIVATION_1_SIZE    (13440)



#define AI_SPEECH_MODEL_DATA_WEIGHTS_SIZES \
  { 176888, }
#define AI_SPEECH_MODEL_DATA_WEIGHTS_SIZE         (176888)
#define AI_SPEECH_MODEL_DATA_WEIGHTS_COUNT        (1)
#define AI_SPEECH_MODEL_DATA_WEIGHT_1_SIZE        (176888)



#define AI_SPEECH_MODEL_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_speech_model_activations_table[1])

extern ai_handle g_speech_model_activations_table[1 + 2];



#define AI_SPEECH_MODEL_DATA_WEIGHTS_TABLE_GET() \
  (&g_speech_model_weights_table[1])

extern ai_handle g_speech_model_weights_table[1 + 2];


#endif    /* SPEECH_MODEL_DATA_PARAMS_H */
