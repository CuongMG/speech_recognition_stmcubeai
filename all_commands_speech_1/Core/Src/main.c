/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "app_x-cube-ai.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "stdlib.h"
#include "string.h"
#include "stdio.h"
#include "arm_math.h"
//#include "math_helper.h"
#include "feature_extraction.h"

#include "speech_model.h"
#include "speech_model_data.h"

#include "i2c-lcd.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
#define SAMPLE_RATE  16000U

/* Number of FFT points. It must be greater or equal to FRAME_LEN */
#define FFT_LEN       2048U

/* Window length and then padded with zeros to match FFT_LEN */
#define FRAME_LEN   FFT_LEN

/* Number of overlapping samples between successive frames */
#define HOP_LEN        512U

/* Number of Mel bands */
#define NUM_MELS       128U

/* Number of Mel filter weights. Returned by MelFilterbank_Init */
#define NUM_MEL_COEFS 2020U

/* Number of MFCCs to return */
#define NUM_MFCC        16U

#define PCM_BUFFER_SIZE 15872
#define DICTIONARY_SIZE 38
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
const char _BACKGROUND_NOISE_[] = "_background_noise_";
const char _SILENCE_[] = "_silence_";
const char _UNKNOWN_[] = "_unknown_";
const char BACKWARD[] = "backward";
const char BED[] = "bed";
const char BIRD[] = "bird";
const char CAT[] = "cat";
const char DOG[] = "dog";
const char DOWN[] = "down";
const char EIGHT[] = "eight";
const char FIVE[] = "five";
const char FOLLOW[] = "follow";
const char FORWARD[] = "forward";
const char FOUR[] = "four";
const char GO[] = "go";
const char HAPPY[] = "happy";
const char HOUSE[] = "house";
const char LEARN[] = "learn";
const char LEFT[] = "left";
const char MARVIN[] = "marvin";
const char NINE[] = "nine";
const char NO[] = "no";
const char OFF[] = "off";
const char ON[] = "on";
const char ONE[] = "one";
const char RIGHT[] = "right";
const char SEVEN[] = "seven";
const char SHEILA[] = "sheila";
const char SIX[] = "six";
const char STOP[] = "stop";
const char THREE[] = "three";
const char TREE[] = "tree";
const char TWO[] = "two";
const char UP[] = "up";
const char VISUAL[] = "visual";
const char WOW[] = "wow";
const char YES[] = "yes";
const char ZERO[] = "zero";

static const char *dictionary[DICTIONARY_SIZE] = { _BACKGROUND_NOISE_,
		_SILENCE_, _UNKNOWN_, BACKWARD, BED, BIRD, CAT, DOG, DOWN, EIGHT, FIVE,
		FOLLOW, FORWARD, FOUR, GO, HAPPY, HOUSE, LEARN, LEFT, MARVIN, NINE, NO,
		OFF, ON, ONE, RIGHT, SEVEN, SHEILA, SIX, STOP, THREE, TREE, TWO, UP,
		VISUAL, WOW, YES, ZERO };

char data[50];
		/* Instance for floating-point RFFT/RIFFT */
arm_rfft_fast_instance_f32 rfft;

/* Instance for the floating-point MelFilterbank function */
MelFilterTypeDef mel_filter;

/* Instance for the floating-point DCT functions */
DCT_InstanceTypeDef dct;

/* Instance for the floating-point Spectrogram function */
SpectrogramTypeDef spectrogram;

/* Instance for the floating-point MelSpectrogram function */
MelSpectrogramTypeDef mel_spectrogram;

/* Instance for the floating-point Log-MelSpectrogram function */
LogMelSpectrogramTypeDef log_mel_spectrogram;

/* Instance for the floating-point Mfcc function */
MfccTypeDef mfcc;

/* Intermediate buffer that contains a signal frame */
float32_t frame_buffer[FRAME_LEN];

/* Intermediate buffer that contains Mel-Frequency Cepstral Coefficients (MFCCs) column  */
float32_t mfcc_col_buffer[NUM_MFCC];

/* Intermediate buffer that contains the window function  */
float32_t window_func_buffer[FRAME_LEN];

/* Temporary calculation buffer of length `FFT_LEN` */
float32_t spectrogram_scratch_buffer[FFT_LEN];

/* Intermediate buffer that contains the Discrete Cosine Transform coefficients */
float32_t dct_coefs_buffer[NUM_MELS * NUM_MFCC];

/* Temporary calculation buffer of length `NUM_MELS` */
float32_t mfcc_scratch_buffer[NUM_MELS];

/*Intermediate buffer that contains the Mel filter weights of length `NUM_MEL_COEFS` */
float32_t mel_filter_coefs[NUM_MEL_COEFS];

/* Intermediate buffer that contains the Mel filter coefficients start indices */
uint32_t mel_filter_start_indices[NUM_MELS];

/* Intermediate buffer that contains the Mel filter coefficients stop indices */
uint32_t mel_filter_stop_indices[NUM_MELS];

/* Number of frames of the input signal */
uint32_t num_frames = 1 + (PCM_BUFFER_SIZE - FRAME_LEN) / HOP_LEN;

volatile uint8_t Inference_Mode = 0; //Indicate if all the data is transferred to be inference
int16_t ADC_buffer[PCM_BUFFER_SIZE];
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
AI_ALIGNED(4) ai_u8 activations[AI_SPEECH_MODEL_DATA_ACTIVATIONS_SIZE];

/* Buffers used to store input and output tensors */
AI_ALIGNED(4) ai_float in_data[AI_SPEECH_MODEL_IN_1_SIZE_BYTES];
AI_ALIGNED(4) ai_float out_data[AI_SPEECH_MODEL_OUT_1_SIZE_BYTES];

ai_buffer *ai_input; // khai bao du lieu dau vao chuan cua mang neral
ai_buffer *ai_output; // khai bao du lieu dau ra chuan cua mang neral
float out_infer, out_infer_stop;
float count =0;
uint32_t time_start=0;
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;
DMA_HandleTypeDef hdma_adc1;

CRC_HandleTypeDef hcrc;

I2C_HandleTypeDef hi2c1;

TIM_HandleTypeDef htim1;

UART_HandleTypeDef huart1;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_ADC1_Init(void);
static void MX_I2C1_Init(void);
static void MX_TIM1_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_CRC_Init(void);
/* USER CODE BEGIN PFP */
void preprocessing_init(void){
	/* Init window function */
	if (Window_Init(window_func_buffer, FRAME_LEN, WINDOW_HANN) != 0) {
		while (1)
			;
	}

	/* Init RFFT */
	arm_rfft_fast_init_f32(&rfft, FFT_LEN);

	/* Init mel filterbank */
	mel_filter.pStartIndices = mel_filter_start_indices;
	mel_filter.pStopIndices = mel_filter_stop_indices;
	mel_filter.pCoefficients = mel_filter_coefs;
	mel_filter.NumMels = NUM_MELS;
	mel_filter.FFTLen = FFT_LEN;
	mel_filter.SampRate = SAMPLE_RATE;
	mel_filter.FMin = 0.0;
	mel_filter.FMax = mel_filter.SampRate / 2.0;
	mel_filter.Formula = MEL_SLANEY;
	mel_filter.Normalize = 1;
	mel_filter.Mel2F = 1;
	MelFilterbank_Init(&mel_filter);
	if (mel_filter.CoefficientsLength != NUM_MEL_COEFS) {
		while (1)
			;
	}

	/* Init DCT operation */
	dct.NumFilters = NUM_MFCC;
	dct.NumInputs = NUM_MELS;
	dct.Type = DCT_TYPE_II_ORTHO;
	dct.RemoveDCTZero = 0;
	dct.pDCTCoefs = dct_coefs_buffer;
	if (DCT_Init(&dct) != 0) {
		while (1)
			;
	}

	/* Init Spectrogram */
	spectrogram.pRfft = &rfft;
	spectrogram.Type = SPECTRUM_TYPE_POWER;
	spectrogram.pWindow = window_func_buffer;
	spectrogram.SampRate = SAMPLE_RATE;
	spectrogram.FrameLen = FRAME_LEN;
	spectrogram.FFTLen = FFT_LEN;
	spectrogram.pScratch = spectrogram_scratch_buffer;

	/* Init MelSpectrogram */
	mel_spectrogram.SpectrogramConf = &spectrogram;
	mel_spectrogram.MelFilter = &mel_filter;

	/* Init LogMelSpectrogram */
	log_mel_spectrogram.MelSpectrogramConf = &mel_spectrogram;
	log_mel_spectrogram.LogFormula = LOGMELSPECTROGRAM_SCALE_DB;
	log_mel_spectrogram.Ref = 1.0;
	log_mel_spectrogram.TopdB = HUGE_VALF;

	/* Init MFCC */
	mfcc.LogMelConf = &log_mel_spectrogram;
	mfcc.pDCT = &dct;
	mfcc.NumMfccCoefs = NUM_MFCC;
	mfcc.pScratch = mfcc_scratch_buffer;
}
void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hadc);
	if (hadc == &hadc1){
	//if (hadc->Instance == ADC1){
	//Bi?n thông báo nh?n d? 16896 d? li?u d? th?c hi?n Inference trên mô hình
		Inference_Mode = 1;
	}
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_ADC_ConvCpltCallback could be implemented in the user file
   */
}
void preprocess_audio(void) {

	for (uint32_t frame_index = 0; frame_index < num_frames; frame_index++) {

		/* Convert 16-bit PCM into normalized floating point values */
		//buf_to_float_normed(&input_signal[HOP_LEN * frame_index], frame_buffer, FRAME_LEN);
		for(uint16_t i = 0; i < 2048; i++){
			frame_buffer[i]=(ADC_buffer[frame_index * HOP_LEN + i]-2047.0)/2047.0;
		}

		MfccColumn(&mfcc, frame_buffer, mfcc_col_buffer);

		/* Reshape column into `out_mfcc` */
		for (uint32_t i = 0; i < NUM_MFCC; i++) {
			in_data[i * num_frames + frame_index] = mfcc_col_buffer[i];
		}
	}
}
uint8_t idx;
ai_float max_f;
void get_word(){
	idx = 0;
	max_f = out_data[0];
	ai_float tmp;

	for (uint8_t i = 1; i < DICTIONARY_SIZE; i++) {
		tmp = out_data[i];
		if (tmp > max_f) {
			max_f = tmp;
			idx = i;
		}
	}
	if(idx == 29){
		HAL_GPIO_TogglePin(led_green_GPIO_Port, led_green_Pin);
		
		lcd_put_cur(1, 1);
		sprintf(data, "accuracy: %.2f", max_f);
		lcd_send_string(&data[0]);
	}
	//char *kq = dictionary[idx];
}
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */
	ai_error err;
	ai_network_report report;	
	ai_handle speech_model = AI_HANDLE_NULL;
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_ADC1_Init();
  MX_I2C1_Init();
  MX_TIM1_Init();
  MX_USART1_UART_Init();
  MX_CRC_Init();
  //MX_X_CUBE_AI_Init();
  /* USER CODE BEGIN 2 */
	//----Khoi tao lcd----------------	
	lcd_init();
	lcd_clear();
	lcd_put_cur(0,3);
	lcd_send_string("Word: stop");
	//---------------------
	preprocessing_init();
	HAL_ADC_Start_DMA(&hadc1, (uint32_t*)ADC_buffer, PCM_BUFFER_SIZE);
	//----Khoi tao model----------------	
	  const ai_handle acts[] = { activations };
    err = ai_speech_model_create_and_init(&speech_model, acts, NULL);  // tao model

    if (err.type != AI_ERROR_NONE) {
        printf("ai init_and_create error\n");
        return -1;
    }
		if (ai_speech_model_get_report(speech_model, &report) != true) {
        printf("ai get report error\n");
        return -1;
    }
//------ In thong tin model ---------------		
		printf("Model name      : %s\n", report.model_name);
    printf("Model signature : %s\n", report.model_signature);

    ai_input = &report.inputs[0];
    ai_output = &report.outputs[0];
    printf("input[0] : (%d, %d, %d)\n", AI_BUFFER_SHAPE_ELEM(ai_input, AI_SHAPE_HEIGHT),
                                        AI_BUFFER_SHAPE_ELEM(ai_input, AI_SHAPE_WIDTH),
                                        AI_BUFFER_SHAPE_ELEM(ai_input, AI_SHAPE_CHANNEL));
    printf("output[0] : (%d, %d, %d)\n", AI_BUFFER_SHAPE_ELEM(ai_output, AI_SHAPE_HEIGHT),
                                         AI_BUFFER_SHAPE_ELEM(ai_output, AI_SHAPE_WIDTH),
                                         AI_BUFFER_SHAPE_ELEM(ai_output, AI_SHAPE_CHANNEL));

//-- Doc cau truc data input, data output cua mang NN	
		ai_i32 n_batch;
		ai_input = ai_speech_model_inputs_get(speech_model, NULL);
    ai_output = ai_speech_model_outputs_get(speech_model, NULL);
		
//---- gan data vao mang NN 		
		ai_input[0].data = AI_HANDLE_PTR(in_data);
    ai_output[0].data = AI_HANDLE_PTR(out_data);
		
//--- Start timer/counter
		HAL_TIM_Base_Start(&htim1);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

  //MX_X_CUBE_AI_Process();
    /* USER CODE BEGIN 3 */
		if(Inference_Mode == 1){			
			preprocess_audio();
			
			/*-----------AI_run_start---------*/			
			
//---Get current timestamp
    time_start = htim1.Instance->CNT;		
//----- Run model	
		n_batch = ai_speech_model_run(speech_model, &ai_input[0], &ai_output[0]);
	  if (n_batch != 1) {
        err = ai_speech_model_get_error(speech_model);
        printf("ai run error %d, %d\n", err.type, err.code);
      return -1;
    }
		out_infer = ((float *)out_data)[0]; // chuyen 4bytes int -> float
		out_infer_stop = ((float *)out_data)[29];
//    for (int i = 0; i < AI_SIN_MODEL_OUT_1_SIZE; i++) {
//        printf("%d", out_data[i]);
//    }
		sprintf(&data[0], "\nInference times : %d", htim1.Instance->CNT - time_start);
		HAL_UART_Transmit(&huart1, (uint8_t *)data, strlen(data), 100);
		
		printf("\nInference times:%d",htim1.Instance->CNT - time_start);
    printf("\nInference output:%f",out_infer);
//---------------------------------------
		//HAL_UART_Transmit(&huart2, M, strlen(M), 1000);
			/*-----------AI_run_stop---------*/
			get_word();
			memset(ADC_buffer, 0, PCM_BUFFER_SIZE * 2);
			Inference_Mode = 0;
			HAL_ADC_Start_DMA(&hadc1, (uint32_t*)ADC_buffer, PCM_BUFFER_SIZE);
		}
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE2);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 123;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */

  /** Configure the global features of the ADC (Clock, Resolution, Data Alignment and number of conversion)
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV8;
  hadc1.Init.Resolution = ADC_RESOLUTION_12B;
  hadc1.Init.ScanConvMode = DISABLE;
  hadc1.Init.ContinuousConvMode = ENABLE;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion = 1;
  hadc1.Init.DMAContinuousRequests = ENABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_0;
  sConfig.Rank = 1;
  sConfig.SamplingTime = ADC_SAMPLETIME_480CYCLES;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

/**
  * @brief CRC Initialization Function
  * @param None
  * @retval None
  */
static void MX_CRC_Init(void)
{

  /* USER CODE BEGIN CRC_Init 0 */

  /* USER CODE END CRC_Init 0 */

  /* USER CODE BEGIN CRC_Init 1 */

  /* USER CODE END CRC_Init 1 */
  hcrc.Instance = CRC;
  if (HAL_CRC_Init(&hcrc) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN CRC_Init 2 */

  /* USER CODE END CRC_Init 2 */

}

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.ClockSpeed = 100000;
  hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Analogue filter
  */
  if (HAL_I2CEx_ConfigAnalogFilter(&hi2c1, I2C_ANALOGFILTER_ENABLE) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Digital filter
  */
  if (HAL_I2CEx_ConfigDigitalFilter(&hi2c1, 0) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 123-1;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 65535;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA2_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA2_Stream0_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream0_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream0_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
/* USER CODE BEGIN MX_GPIO_Init_1 */
/* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOG, led_green_Pin|led_red_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pins : button1_Pin button2_Pin */
  GPIO_InitStruct.Pin = button1_Pin|button2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : led_green_Pin led_red_Pin */
  GPIO_InitStruct.Pin = led_green_Pin|led_red_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

/* USER CODE BEGIN MX_GPIO_Init_2 */
/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
