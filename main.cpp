/**
 * Main file that does the actual upscaling of an image.
 * Spit out a binary file which can be parsed to generate the neural network
 **/
#include <bits/stdc++.h>

#include "include/external/stb_image.h"
#include "include/neural_network.hpp"
#include "raylib.h"
using namespace functions;
using namespace std;

#define PRINT_PARSED_INFO
#define FPS 1500
#define RATE 1e-2
#define EPSILLON 1e-2
#define EPOCHS (50 * 1000)
const int screenWidth = 16 * 100;
const int screenHeight = 9 * 100;

// Dark theme background color
Color backgroundColor = {0x18, 0x18, 0x18, 0xFF};

const char *imagePaths[5] = {"mnist/train/100.png", "mnist/train/200.png",
                             "mnist/train/300.png", "mnist/train/400.png",
                             "mnist/train/500.png"};

const char *imageNames[5] = {"MNIST Digit 100", "MNIST Digit 200",
                             "MNIST Digit 300", "MNIST Digit 400",
                             "MNIST Digit 500"};

int currentImageIndex = 0; // Default to index 0 (100.png)

bool imageLoadedSuccessfully = true;
string imageLoadErrorFilename = "";

bool invalidName(string s) {
  return s.size() < 4 || s.substr(s.size() - 4, 4) != ".mat";
}

void displayImage(NeuralNetwork &nn, unsigned imgHeight) {
  ofstream image("image.txt");
  for (unsigned y = 0; y < imgHeight; y++) {
    for (unsigned x = 0; x < imgHeight; x++) {
      nn.input().value(0, 0) = (float)x / (imgHeight - 1);
      nn.input().value(0, 1) = (float)y / (imgHeight - 1);
      nn.forward();
      unsigned pixel = 255 * nn.output().value(0, 0);
      image << setw(4) << pixel;
    }
    image << endl;
  }
}

typedef enum ErrorNo {
  INPUT_NOT_GIVEN,
  COULD_NOT_READ_IMAGE,
  IMAGE_AINT_8_BITS,
} ErrorNum;

namespace parse {
vector<float (*)(float)> parseActivationFunctions(string filepath) {
  vector<float (*)(float)> activationFunctions;
  ifstream file(filepath);

  if (!file.is_open()) {
    cout << "Could not open activation functions file" << endl;
    return activationFunctions;
  }

  string functionName;
  while (file >> functionName) {
    if (functionName == "ReLU") {
      activationFunctions.push_back(functions::ReLU);
    } else if (functionName == "sigmoidf") {
      activationFunctions.push_back(functions::sigmoidf);
    }
  }

  file.close();
  cout << "\e[32mActivation functions parsed successfully\e[0m" << endl;
  return activationFunctions;
}

vector<unsigned> parseArchitecture(string filepath) {
  ifstream file(filepath);
  if (!file.is_open()) {
    cout << "Could not open file" << endl;
    return {};
  }

  vector<unsigned> architecture;
  unsigned value;
  while (file >> value) {
    architecture.push_back(value);
  }

  file.close();
  cout << "\e[32mArchitecture parsed successfully\e[0m" << endl;
  return architecture;
}
}; // namespace parse

// Load an original image as a texture with point filtering
Texture2D loadOriginalTexture(const char *filepath) {
  Image img = LoadImage(filepath);
  Texture2D tex = LoadTextureFromImage(img);
  SetTextureFilter(tex, TEXTURE_FILTER_POINT);
  UnloadImage(img);
  return tex;
}

// Load a 28x28 PNG and populate training data matrix in-place
bool tryLoadImage(int index, matrix<> &trainingData, int &imgWidth,
                  int &imgHeight, Texture2D &originalTexture) {
  if (index < 0 || index >= 5)
    return false;
  const char *filepath = imagePaths[index];

  if (!FileExists(filepath)) {
    imageLoadedSuccessfully = false;
    imageLoadErrorFilename = filepath;
    return false;
  }

  int imgComponents;
  unsigned char *img_pixels = (unsigned char *)stbi_load(
      filepath, &imgWidth, &imgHeight, &imgComponents, 1);
  if (img_pixels == NULL) {
    imageLoadedSuccessfully = false;
    imageLoadErrorFilename = filepath;
    return false;
  }
  if (imgWidth != 28 || imgHeight != 28) {
    cerr << "Error: Only 28x28 images are supported. Got " << imgWidth << "x"
         << imgHeight << "\n";
    stbi_image_free(img_pixels);
    imageLoadedSuccessfully = false;
    imageLoadErrorFilename = filepath;
    return false;
  }

  for (int y = 0; y < imgHeight; y++) {
    for (int x = 0; x < imgWidth; x++) {
      int i = y * imgWidth + x;
      float normalized_x = float(x) / (imgWidth - 1);
      float normalized_y = float(y) / (imgHeight - 1);
      float normalized_brightness = (float)img_pixels[i] / 255.0f;

      trainingData.value(i, 0) = normalized_x;
      trainingData.value(i, 1) = normalized_y;
      trainingData.value(i, 2) = normalized_brightness;
    }
  }
  stbi_image_free(img_pixels);

  // If raylib has initialized the graphics context, reload texture
  if (IsWindowReady()) {
    if (originalTexture.id != 0) {
      UnloadTexture(originalTexture);
      originalTexture.id = 0;
    }
    originalTexture = loadOriginalTexture(filepath);
  }

  imageLoadedSuccessfully = true;
  imageLoadErrorFilename = "";
  return true;
}

// Reset network weights and training state
void resetTraining(NeuralNetwork &nn, int &epoch, bool &pause) {
  srand(time(0));
  nn.randomise(-1, 1);
  epoch = 1;
}

// Switch current image target and restart training
void switchImage(int index, matrix<> &trainingData, int &imgWidth,
                 int &imgHeight, Texture2D &originalTexture, NeuralNetwork &nn,
                 int &epoch, bool &pause) {
  if (index < 0 || index >= 5)
    return;
  currentImageIndex = index;
  if (tryLoadImage(currentImageIndex, trainingData, imgWidth, imgHeight,
                   originalTexture)) {
    resetTraining(nn, epoch, pause);
    pause = false; // Auto unpause
  } else {
    epoch = 1;
    pause = true; // Pause training if image failed to load
  }
}

// Generate high-resolution reconstruction (upscaled version) using the NN model
void updateReconstructionTexture(NeuralNetwork &nn, Texture2D &reconTexture,
                                 std::vector<Color> &reconPixels, int size) {
  for (int y = 0; y < size; y++) {
    for (int x = 0; x < size; x++) {
      nn.input().value(0, 0) = (float)x / (size - 1);
      nn.input().value(0, 1) = (float)y / (size - 1);
      nn.forward();
      float val = nn.output().value(0, 0);
      if (val < 0.0f)
        val = 0.0f;
      if (val > 1.0f)
        val = 1.0f;
      unsigned char pixelVal = (unsigned char)(255.0f * val);
      reconPixels[y * size + x] = (Color){pixelVal, pixelVal, pixelVal, 255};
    }
  }
  UpdateTexture(reconTexture, reconPixels.data());
}

void NN_render_raylib(NeuralNetwork &nn, const vector<unsigned> &arch, int rx,
                      int ry, int rw, int rh) {
  Color lowColor = RED;
  Color highColor = DARKBLUE;

  float neuronRadius = rh * 0.03;
  int layer_border_vpad = rh * 0.08;
  int layer_border_hpad = rw * 0.06;
  int nn_width = rw - 2 * layer_border_hpad;
  int layer_hpad = nn_width / arch.size();
  int nn_height = rh - 2 * layer_border_vpad;
  int nn_x = rx + rw / 2 - nn_width / 2;
  int nn_y = ry + rh / 2 - nn_height / 2;
  for (unsigned l = 0; l < arch.size(); l++) {
    int layer_vpad1 = nn_height / (arch[l]);
    for (unsigned j = 0; j < arch[l]; j++) {
      int cx1 = nn_x + l * layer_hpad + layer_hpad / 2;
      int cy1 = nn_y + j * layer_vpad1 + layer_vpad1 / 2;
      if (l + 1 < arch.size()) {
        for (unsigned k = 0; k < arch[l + 1]; k++) {
          int layer_vpad2 = nn_height / (arch[l + 1]);
          int cx2 = nn_x + (l + 1) * layer_hpad + layer_hpad / 2;
          int cy2 = nn_y + k * layer_vpad2 + layer_vpad2 / 2;
          Vector2 start = {(float)cx1, (float)cy1};
          Vector2 end = {(float)cx2, (float)cy2};
          float value = sigmoidf(nn.value("weights", l, j, k));
          highColor.a = floorf(255.0F * value);
          float thickness =
              rh * 0.015F * (sigmoidf(abs(nn.value("weights", l, j, k))) - 0.5);
          DrawLineEx(start, end, thickness,
                     ColorAlphaBlend(lowColor, highColor, WHITE));
        }
      }
      if (l > 0) {
        highColor.a = floor(255.0F * sigmoidf(nn.value("biases", l - 1, 0, j)));
        DrawCircle(cx1, cy1, neuronRadius,
                   ColorAlphaBlend(lowColor, highColor, WHITE));
      } else
        DrawCircle(cx1, cy1, neuronRadius, GRAY);
    }
  }
}

// need to write own validate for each problem
void validate(NeuralNetwork &nn, matrix<> &ti, matrix<> &to) {
  int cnt = 0;
  for (unsigned i = 0; i < ti.getRows(); i++) {
    nn.input() = mat_row(ti, i);
    nn.forward();
    matrix<> output = nn.output();
    matrix<> expected = mat_row(to, i);
    output.apply(round);
    expected.apply(round);
    if (output.value(0, 0) != expected.value(0, 0))
      cout << ti.value(i, 0) << " ^ " << ti.value(i, 1) << " = "
           << nn.output().value(0, 0) << endl;
    if (output == expected)
      cnt++;
  }

  float accuracy = (float)cnt / ti.getRows() * 100;
  cout << "accuracy: " << accuracy << "%" << std::endl;
}

int main(int argc, char *argv[]) {
  // If an image argument is provided, map it to the first slot (index 0)
  if (argc >= 2) {
    imagePaths[0] = argv[1];
    imageNames[0] = argv[1];
  }

  // Default configuration files
  const char *archFile = (argc >= 3) ? argv[2] : "./network.arch";
  const char *funcFile = (argc >= 4) ? argv[3] : "./layers.functions";
  const char *storeLocation = (argc >= 5) ? argv[4] : "img.mat";

  int imgWidth = 28;
  int imgHeight = 28;

  // Allocate matrix with 784 rows and 3 columns (x, y, intensity)
  matrix<> trainingData(imgWidth * imgHeight, 3);

  // Try loading initial target image
  Texture2D originalTexture;
  originalTexture.id = 0;
  tryLoadImage(currentImageIndex, trainingData, imgWidth, imgHeight,
               originalTexture);

  // Save initial matrix to file as done in original code if loaded
  // successfully, but non-interactively
  if (imageLoadedSuccessfully) {
    trainingData.save(storeLocation);
    cout << "\e[32m[INFO] Generated " << storeLocation << " from "
         << imagePaths[currentImageIndex] << "!\n\e[0m";
  }

  // Parse network architecture
  vector<unsigned> arch = parse::parseArchitecture(archFile);
  if (arch.size() < 3) {
    fprintf(stderr,
            "\e[31m<GYM> Architecture does not contain hidden layers?\n\e[0m");
    return 1;
  }

  // Parse activation functions
  vector<float (*)(float)> acFs;
  acFs = parse::parseActivationFunctions(funcFile);

  // Dimensions check
  if (arch[0] + arch.back() != trainingData.getCols()) {
    fprintf(stderr,
            "\e[31m<Check> architecture and data's dimensions do not match!\n"
            "architecture[0]    = %d\n"
            "architecture[last] = %d\n"
            "trainingData.cols  = %d\e[0m",
            arch[0], arch.back(), trainingData.getCols());
    return 1;
  }

  // Prepare ti and to views pointing to trainingData
  matrix<> ti(trainingData.getRows(), arch[0], trainingData.getCols(),
              &trainingData.value(0, 0));
  matrix<> to(trainingData.getRows(), arch.back(), trainingData.getCols(),
              &trainingData.value(0, arch[0]));

  // Initialize raylib window
  SetConfigFlags(FLAG_WINDOW_RESIZABLE);
  InitWindow(screenWidth, screenHeight, "C++ Neural Network Image Upscaler");
  SetTargetFPS(FPS);

  // Load custom modern font
  Font font = LoadFontEx("resources/font.ttf", 32, 0, 250);

  // Load texture of the original image if loaded successfully
  if (imageLoadedSuccessfully) {
    originalTexture = loadOriginalTexture(imagePaths[currentImageIndex]);
  }

  // Setup high-resolution reconstructed texture (224x224 for 8x upscale)
  int reconSize = 224;
  std::vector<Color> reconPixels(reconSize * reconSize, BLACK);
  Image reconImg = GenImageColor(reconSize, reconSize, BLACK);
  Texture2D reconTexture = LoadTextureFromImage(reconImg);
  SetTextureFilter(reconTexture, TEXTURE_FILTER_POINT);
  UnloadImage(reconImg);

  // Initialize neural network weights and gradients
  srand(time(0));
  NeuralNetwork nn, g;
  nn.init(arch, acFs);
  g.init(arch, acFs);
  nn.randomise(-1, 1);

  int epoch = 1;
  bool pause = true;
  float lastCost = 0.0f;

  // Force first cost evaluation and reconstruction update if loaded
  // successfully
  if (imageLoadedSuccessfully) {
    lastCost = nn.cost(ti, to);
    updateReconstructionTexture(nn, reconTexture, reconPixels, reconSize);
  }

  while (!WindowShouldClose()) {
    // Key controls for target image selection
    if (IsKeyPressed(KEY_ONE))
      switchImage(0, trainingData, imgWidth, imgHeight, originalTexture, nn,
                  epoch, pause);
    if (IsKeyPressed(KEY_TWO))
      switchImage(1, trainingData, imgWidth, imgHeight, originalTexture, nn,
                  epoch, pause);
    if (IsKeyPressed(KEY_THREE))
      switchImage(2, trainingData, imgWidth, imgHeight, originalTexture, nn,
                  epoch, pause);
    if (IsKeyPressed(KEY_FOUR))
      switchImage(3, trainingData, imgWidth, imgHeight, originalTexture, nn,
                  epoch, pause);
    if (IsKeyPressed(KEY_FIVE))
      switchImage(4, trainingData, imgWidth, imgHeight, originalTexture, nn,
                  epoch, pause);

    // Key controls for training manipulation
    if (IsKeyPressed(KEY_R)) {
      resetTraining(nn, epoch, pause);
      pause = false; // Start training immediately
    }
    if (IsKeyPressed(KEY_SPACE)) {
      pause = !pause;
    }

    // Training loop step
    bool isTraining = imageLoadedSuccessfully && (epoch <= EPOCHS) && !pause;
    bool needUpdateRecon = false;

    if (isTraining) {
      nn.finite_diff(g, EPSILLON, ti, to);
      nn.learn(g, RATE);

      if (epoch % 50 == 0) {
        lastCost = nn.cost(ti, to);
      }
      if (epoch % 100 == 0) {
        needUpdateRecon = true;
      }
      epoch++;
    }

    // Detect pause/stop transitions and update reconstruction
    static bool wasTraining = false;
    if (!isTraining && wasTraining) {
      if (imageLoadedSuccessfully) {
        lastCost = nn.cost(ti, to);
        needUpdateRecon = true;
      }
    }
    wasTraining = isTraining;

    // Detect new training start/reset and update reconstruction
    if (epoch == 1 && !needUpdateRecon) {
      if (imageLoadedSuccessfully) {
        lastCost = nn.cost(ti, to);
        needUpdateRecon = true;
      }
    }

    if (needUpdateRecon && imageLoadedSuccessfully) {
      updateReconstructionTexture(nn, reconTexture, reconPixels, reconSize);
    }

    // Drawing frame
    BeginDrawing();
    ClearBackground(backgroundColor);
    {
      // 1. Render Neural Network Architecture Diagram (Left)
      NN_render_raylib(nn, arch, 50, 120, 700, 700);

      // 2. Render Top Header Section
      DrawTextEx(font, "NEURAL NETWORK IMAGE UPSCALER", (Vector2){50.0f, 40.0f},
                 28.0f, 1.0f, LIGHTGRAY);

      // Determine status text and colors
      const char *statusStr = "TRAINING...";
      Color badgeBg = ORANGE;
      Color badgeText = BLACK;
      if (!imageLoadedSuccessfully) {
        statusStr = "ERROR";
        badgeBg = RED;
        badgeText = WHITE;
      } else if (epoch > EPOCHS) {
        statusStr = "COMPLETE";
        badgeBg = (Color){46, 204, 113, 255}; // Emerald Green
        badgeText = WHITE;
      } else if (pause) {
        statusStr = "PAUSED";
        badgeBg = DARKGRAY;
        badgeText = WHITE;
      }

      // Draw status pill badge
      DrawRectangleRounded((Rectangle){550, 40, 160, 36}, 0.5f, 4, badgeBg);
      int statusTextWidth = (int)MeasureTextEx(font, statusStr, 16, 1.0f).x;
      DrawTextEx(font, statusStr,
                 (Vector2){550.0f + 80.0f - statusTextWidth / 2.0f, 50.0f},
                 16.0f, 1.0f, badgeText);

      // 3. Render Image Comparison Panel (Right)
      DrawTextEx(font, "ORIGINAL IMAGE (28x28)", (Vector2){900.0f, 140.0f},
                 18.0f, 1.0f, LIGHTGRAY);
      DrawTextEx(font, "NN RECONSTRUCTION (56x56)", (Vector2){1200.0f, 140.0f},
                 18.0f, 1.0f, LIGHTGRAY);

      // Background panels for images
      DrawRectangle(895, 175, 270, 270, (Color){30, 30, 30, 255});
      DrawRectangle(1195, 175, 270, 270, (Color){30, 30, 30, 255});

      if (imageLoadedSuccessfully) {
        // Draw original image visually scaled up to 260x260
        DrawTexturePro(originalTexture,
                       (Rectangle){0.0f, 0.0f, (float)originalTexture.width,
                                   (float)originalTexture.height},
                       (Rectangle){900.0f, 180.0f, 260.0f, 260.0f},
                       (Vector2){0.0f, 0.0f}, 0.0f, WHITE);

        // Draw reconstructed upscaled image visually scaled to 260x260
        DrawTexturePro(reconTexture,
                       (Rectangle){0.0f, 0.0f, (float)reconTexture.width,
                                   (float)reconTexture.height},
                       (Rectangle){1200.0f, 180.0f, 260.0f, 260.0f},
                       (Vector2){0.0f, 0.0f}, 0.0f, WHITE);
      } else {
        // Draw error message where the images would be
        DrawRectangle(895, 175, 570, 270, (Color){40, 20, 20, 255});
        DrawRectangleLines(895, 175, 570, 270, RED);

        string errText = "Image not found: " + imageLoadErrorFilename;
        int textW = (int)MeasureTextEx(font, errText.c_str(), 18, 1.0f).x;
        DrawTextEx(font, errText.c_str(),
                   (Vector2){895.0f + 285.0f - textW / 2.0f, 280.0f}, 18.0f,
                   1.0f, RED);
        int msgW = (int)MeasureTextEx(
                       font, "Please place the MNIST PNG files in mnist/train/",
                       14, 1.0f)
                       .x;
        DrawTextEx(font, "Please place the MNIST PNG files in mnist/train/",
                   (Vector2){895.0f + 285.0f - msgW / 2.0f, 320.0f}, 14.0f,
                   1.0f, LIGHTGRAY);
      }

      // 4. Render Info / Stats Box
      DrawRectangleRounded((Rectangle){900, 480, 560, 150}, 0.1f, 4,
                           (Color){28, 28, 28, 255});
      DrawRectangleRoundedLines((Rectangle){900, 480, 560, 150}, 0.1f, 4,
                                (Color){50, 50, 50, 255});

      string epocStr =
          "Epoch: " + to_string(epoch > EPOCHS ? EPOCHS : epoch - 1) + " / " +
          to_string(EPOCHS);
      DrawTextEx(font, epocStr.c_str(), (Vector2){920.0f, 500.0f}, 18.0f, 1.0f,
                 WHITE);

      string costStr = "Loss/Cost: " + to_string(lastCost);
      DrawTextEx(font, costStr.c_str(), (Vector2){920.0f, 530.0f}, 18.0f, 1.0f,
                 WHITE);

      string targetStr = "Active Target: Image " +
                         to_string(currentImageIndex + 1) + "/5 (" +
                         imageNames[currentImageIndex] + ")";
      DrawTextEx(font, targetStr.c_str(), (Vector2){920.0f, 560.0f}, 18.0f,
                 1.0f, SKYBLUE);

      // Sleek Progress Bar
      float progress = (float)(epoch - 1) / EPOCHS;
      if (progress > 1.0f)
        progress = 1.0f;
      DrawRectangle(920, 595, 520, 12, (Color){45, 45, 45, 255});
      DrawRectangle(920, 595, (int)(520 * progress), 12, SKYBLUE);

      // 5. Render Controls Reminder Box
      DrawRectangleRounded((Rectangle){900, 650, 560, 130}, 0.1f, 4,
                           (Color){28, 28, 28, 255});
      DrawRectangleRoundedLines((Rectangle){900, 650, 560, 130}, 0.1f, 4,
                                (Color){50, 50, 50, 255});
      DrawTextEx(font, "CONTROLS REMINDER", (Vector2){920.0f, 665.0f}, 16.0f,
                 1.0f, SKYBLUE);
      DrawTextEx(font, "[SPACE] : Pause / Resume Training",
                 (Vector2){920.0f, 695.0f}, 14.0f, 1.0f, LIGHTGRAY);
      DrawTextEx(font, "[R]     : Reset Network Weights & Restart",
                 (Vector2){920.0f, 715.0f}, 14.0f, 1.0f, LIGHTGRAY);
      DrawTextEx(font, "[1] - [5] : Switch Target Images (Reset & Train)",
                 (Vector2){920.0f, 735.0f}, 14.0f, 1.0f, LIGHTGRAY);

      // Render FPS
      string fpsStr = "FPS: " + to_string(GetFPS());
      int fpsW = (int)MeasureTextEx(font, fpsStr.c_str(), 18, 1.0f).x;
      DrawTextEx(font, fpsStr.c_str(),
                 (Vector2){(float)(screenWidth - fpsW - 50), 48.0f}, 18.0f,
                 1.0f, GRAY);
    }
    EndDrawing();
  }

  // Cleanup resources
  UnloadTexture(originalTexture);
  UnloadTexture(reconTexture);
  UnloadFont(font);
  CloseWindow();

  return 0;
}