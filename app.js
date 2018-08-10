async function main() {
    let model;
    const video = document.getElementById("webcam");
    const mobilenet = await loadModel()
    const webcam = new Webcam(video)
    webcam.setup()

    happyButton.addEventListener("mousedown", () => handle(0))
    happyButton.addEventListener("mouseup", () => mouseDown = false)

    sadButton.addEventListener("mousedown", () => handle(1))
    sadButton.addEventListener("mouseup", () => mouseDown = false)

    trainButton.addEventListener("click", () => train())
    predictButton.addEventListener("click", () => predict())



    async function handle(label) {
        mouseDown = true;
        const img = webcam.capture()
        while (mouseDown) {
            controllerDataset.addExample(mobilenet.predict(img), label)
            if (label == 0) {
                happyFaceCount++;
                happyButton.innerHTML = `Happy Face ${happyFaceCount}`
            }

            if (label == 1) {
                sadFaceCount++;
                sadButton.innerHTML = `Sad Face ${sadFaceCount}`
            }

            if (happyFaceCount > 0 && sadFaceCount > 0) {
                trainButton.disabled = false;
            }

            await tf.nextFrame();
        }


    }

    async function train() {
        if (controllerDataset.xs == null) {
            console.log("No Data to be trained on")
        }

        model = tf.sequential({
            layers: [
                tf.layers.flatten({
                    inputShape: [7, 7, 256]
                }),
                tf.layers.dense({
                    units: 100,
                    activation: 'relu',
                    kernelInitializer: 'varianceScaling',
                    useBias: true
                }),
                tf.layers.dense({
                    units: 2,
                    kernelInitializer: 'varianceScaling',
                    useBias: false,
                    activation: 'softmax'
                })
            ]
        })

        const optimizer = tf.train.adam(0.0001);

        model.compile({
            optimizer: optimizer,
            loss: 'categoricalCrossentropy'
        });

        const batchSize = Math.floor(controllerDataset.xs.shape[0] * 0.4);
        if (!(batchSize > 0)) {
            throw new Error(
                `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
        }

        model.fit(controllerDataset.xs, controllerDataset.ys, {
            batchSize,
            epochs: 20,
            callbacks: {
                onBatchEnd: async (batch, logs) => {
                    //console.log('Loss: ' + logs.loss.toFixed(5));
                    await tf.nextFrame();
                },
                onTrainEnd: () => {
                    predictButton.disabled = false;
                }
            }
        });
    }

    async function predict() {
        const predictionClass = tf.tidy(() => {
            const img = webcam.capture();

            const activation = mobilenet.predict(img);

            const prediction = model.predict(activation);

            return prediction.as1D().argMax();
        })

        const classID = (await predictionClass.data())[0];
        predictionClass.dispose();
        emotionText.innerHTML = `Current Emotion : ${EMOTIONS[classID]}`
        await tf.nextFrame();
        requestAnimationFrame(predict);
    }
}


main();