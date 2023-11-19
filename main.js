function computeColorForLabels(classID){
    if(classID == 0){
        color=[85, 45, 255, 255]
    }
    else if(classID == 2){
        color=[222, 82, 175, 255]
    }
    else if(classID == 3){
        color=[0, 204, 255, 255]
    }
    else if(classID == 4){
        color = [0, 149, 255, 255]
    }
    else{
        color = [255,111,111,255]
    }
    return color;
}
function handleImageInput(event){
    const fileInput = event.target;
    const file = fileInput.files[0];
    if (file){
        const reader = new FileReader();
        reader.onload = function (e) {
            const imgMain = document.getElementById("img-main");
            imgMain.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
    
}
function downloadImage() {
    // Get the canvas element
    const canvas = document.getElementById('main-canvas');

    // Create an anchor element to trigger the download
    const link = document.createElement('a');

    // Set the download attribute with a filename (you can customize the filename)
    link.download = 'objects_detection.png';

    // Convert the canvas content to a data URL
    const dataUrl = canvas.toDataURL();

    // Set the href attribute of the anchor with the data URL
    link.href = dataUrl;

    // Append the anchor to the document
    document.body.appendChild(link);

    // Trigger a click on the anchor element to start the download
    link.click();

    // Remove the anchor element from the document
    document.body.removeChild(link);
} 
function opencvReady(){
    cv["onRuntimeInitialized"] = () =>
    {
        console.log("OpenCV Ready");
        const labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light","fire hydrant", "stop sign", "parking meter", "bench", "bird","cat", "dog", "horse", "sheep", "cow","elephant", "bear", "zebra", "giraffe", "backpack","umbrella", "handbag", "tie", "suitcase", "frisbee","skis", "snowboard", "sports ball", "kite", "baseball bat","baseball glove", "skateboard", "surfboard", "tennis racket", "bottle","wine glass", "cup", "fork", "knife", "spoon","bowl", "banana", "apple", "sandwich", "orange","broccoli", "carrot", "hot dog", "pizza", "donut","cake", "chair", "couch", "potted plant", "bed","dining table", "toilet", "tv", "laptop", "mouse","remote", "keyboard", "cell phone", "microwave", "oven","toaster", "sink", "refrigerator", "book", "clock","vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        const numClass = 80;
        //Read an Image from the image source and covert it into the OpenCV Format
        //In JS we use Let to define our variable
        let imgMain = cv.imread("img-main");
        cv.imshow("main-canvas", imgMain);
        imgMain.delete();
        //Image Upload
        document.getElementById("image-upload").addEventListener("change", handleImageInput);
        //RGB Image
        document.getElementById("RGB-Image").onclick = function(){
            console.log("RGB Image");
            let imgMain = cv.imread("img-main");
            cv.imshow("main-canvas", imgMain);
            imgMain.delete();
        };

        //Gray Scale Image
        document.getElementById("Gray-Scale-Image").onclick = function(){
            console.log("Gray Scale Image");
            let imgMain = cv.imread("img-main");
            let imgGray = new cv.Mat();
            cv.cvtColor(imgMain, imgGray, cv.COLOR_RGBA2GRAY);
            cv.imshow("main-canvas", imgGray);
            imgMain.delete();
            imgGray.delete();
        };

        //Object Detection Image
        document.getElementById("Object-Detection-Image").onclick = async function(){
            console.log("Object Detection Image");
            // const labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light","fire hydrant", "stop sign", "parking meter", "bench", "bird","cat", "dog", "horse", "sheep", "cow","elephant", "bear", "zebra", "giraffe", "backpack","umbrella", "handbag", "tie", "suitcase", "frisbee","skis", "snowboard", "sports ball", "kite", "baseball bat","baseball glove", "skateboard", "surfboard", "tennis racket", "bottle","wine glass", "cup", "fork", "knife", "spoon","bowl", "banana", "apple", "sandwich", "orange","broccoli", "carrot", "hot dog", "pizza", "donut","cake", "chair", "couch", "potted plant", "bed","dining table", "toilet", "tv", "laptop", "mouse","remote", "keyboard", "cell phone", "microwave", "oven","toaster", "sink", "refrigerator", "book", "clock","vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
            // const numClass = 80;
            let image = document.getElementById("img-main");
            let inputImage = cv.imread(image);
            console.log("Image Shape", inputImage.rows, inputImage.cols);
            // Load the Tensorflow js Model
            let model = await tf.loadGraphModel("yolov8n_web_model/model.json");
            const inputTensorShape = model.inputs[0].shape;
            const modelWidth = inputTensorShape[1];
            const modelHeight  = inputTensorShape[2];
            console.log("Model Width", modelWidth, "Model Height", modelHeight);
            const preprocess = (image, modelWidth, modelHeight) => {
                let xRatio, yRatio;
                const input = tf.tidy(()=>{
                //Convert the Pixel Data From an Image Source into a TensorFlow js Tensor
                const img = tf.browser.fromPixels(image);
                //Extracting the height and Width of the Image Tensor
                const [h,w] = img.shape.slice(0,2);
                //Height and Width of the Image Tensor
                console.log("Height", h, "Width", w)
                //Calculate the Maximum Value between Width and Height
                const maxSize = Math.max(w, h);
                const imgPadded = img.pad([
                    [0, maxSize - h],
                    [0, maxSize  - w], 
                    [0,0]
                ]);
                xRatio = maxSize/w;
                yRatio = maxSize/h;
                return tf.image.resizeBilinear(imgPadded, [modelWidth, modelHeight])
                .div(255.0)
                .expandDims(0)
                });
            return [input, xRatio, yRatio]
            };
            const [input, xRatio, yRatio] = preprocess(image, modelWidth, modelHeight);
            console.log("Input Shape", input.shape, "X-Ratio", xRatio, "Y-Ratio", yRatio);

            const res = model.execute(input);
            const transRes = res.transpose([0,2,1]);
            const boxes = tf.tidy(()=>{
                const w = transRes.slice([0,0,2], [-1, -1, 1]);
                const h = transRes.slice([0,0,3], [-1, -1, 1]);
                const x1 = tf.sub(transRes.slice([0,0,0], [-1, -1, 1]), tf.div(w, 2));
                const y1 = tf.sub(transRes.slice([0,0,1], [-1, -1, 1]), tf.div(h, 2));
                return tf.concat([y1, x1, tf.add(y1, h), tf.add(x1, w)], 2).squeeze();
            });
            const [scores, classes]= tf.tidy(() =>{
                const rawScores = transRes.slice([0,0,4], [-1,-1,numClass]).squeeze(0);
                return [rawScores.max(1), rawScores.argMax(1)]; 
            });
            //Apply Non Max Suppression
            const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.2);
            const predictionsLength = nms.size;
            console.log("Prediction Length", predictionsLength)
            
            if (predictionsLength > 0){
                const boxes_data = boxes.gather(nms, 0).dataSync();
                const scores_data = scores.gather(nms, 0).dataSync();
                const classes_data = classes.gather(nms, 0).dataSync();
                console.log("Boxes Data", boxes_data, "Score Data", scores_data, "Classes Data", classes_data); 
                const xScale = inputImage.cols/modelWidth;
                const yScale = inputImage.rows/modelHeight;
                console.log("Score Data Length", scores_data.length);
                for (let i=0; i < scores_data.length; ++i){
                    const classID = classes_data[i];
                    console.log(classID)
                    const className = labels[classes_data[i]];
                    const confidenceScore = (scores_data[i] * 100).toFixed(1);
                    let [y1, x1, y2, x2] = boxes_data.slice(i*4, (i+1)*4);
                    x1 *= xRatio * xScale;
                    x2 *= xRatio * xScale;
                    y1 *= yRatio * yScale;
                    y2 *= yRatio * yScale;
                    const width = x2 - x1;
                    const height = y2 - y1;
                    console.log(x1, y1, width, height, className, confidenceScore);
                    let point1 = new cv.Point(x1, y1);
                    let point2 = new cv.Point(x1+ width, y1 + height);
                    cv.rectangle(inputImage, point1, point2, computeColorForLabels(classID), 4);
                    //const text = `${className} - ${Math.round(confidenceScore)/100}`
                    const text = className + " - " + confidenceScore + "%"
                    // Create a hidden canvas element to measure the text size
                    const canvas = document.createElement("canvas");
                    const context = canvas.getContext("2d");
                    context.font = "22px Arial"; // Set the font size and family as needed
                    // Measure the width of the text
                    const textWidth = context.measureText(text).width;
                    cv.rectangle(inputImage, new cv.Point(x1,y1-20), new cv.Point(x1+ textWidth + context.lineWidth, y1), computeColorForLabels(classID),-1)
                    cv.putText(inputImage, text, new cv.Point(x1, y1 - 5),cv.FONT_HERSHEY_TRIPLEX, 0.50, new cv.Scalar(255,255,255,255), 1);
                }
                cv.imshow("main-canvas", inputImage);
            }
            else{
                cv.imshow("main-canvas", inputImage);
            }
            tf.dispose([res, transRes, boxes, scores, classes, nms]);            
        }

        //Download Image
        document.getElementById("download-image").addEventListener('click', downloadImage);
 
        //Object Detection on Live Webcam Feed
        document.getElementById("live-webcam-feed").onclick = function(){
            console.log("Object Detection on Live Webcam Feed");
            const liveFeed = document.getElementById("live-webcam");
            const enableWebcamButton = document.getElementById("play-pause-webcam");
            let model = undefined;
            let streaming = false;
            let src;
            let cap;

            if (!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)){
                enableWebcamButton.addEventListener('click', ()=>{
                    if(!streaming){
                        console.log("Streaming Started");
                        enableCam();
                        streaming=true;
                    }
                    else{
                        console.log("Streaming Paused");
                        liveFeed.pause();
                        liveFeed.srcObject.getTracks().forEach(track => track.stop());
                        liveFeed.srcObject = null;
                        streaming=false;

                    }
                })
            }
            else{
                console.warn('getUserMedia() is not supported by your browser');

            }

            function enableCam(){
                if(!model){
                    return;
                }
                navigator.mediaDevices.getUserMedia({'video':true, 'audio':false}).then(function(stream){
                    liveFeed.srcObject=stream;
                    liveFeed.addEventListener('loadeddata', predictWebcam);
                })
            }
            setTimeout(async function(){
                try{
                    model = await tf.loadGraphModel("yolov8n_web_model/model.json");
                }
                catch(error){
                    console.log("Error loading YOLOv8 tf.js model")
                }
            }, 0);

            async function predictWebcam(){
                //Check if the Video Element has Loaded the Data
                if(!liveFeed || !liveFeed.videoWidth || !liveFeed.videoHeight){
                    return;
                }
                //Width ---> 640, Height --> 480
                //console.log("Video Width", liveFeed.videoWidth, "Video Height", liveFeed.videoHeight);
                //Width ---> 740, Height --> 540
                console.log("Video Width", liveFeed.width, "Video Height", liveFeed.height);
                const begin = Date.now(); 
                src = new cv.Mat(liveFeed.height, liveFeed.width, cv.CV_8UC4);
                cap = new cv.VideoCapture(liveFeed);
                cap.read(src);
                console.log("Width and Height Defined", src.cols, src.rows);
                const inputTensorShape = model.inputs[0].shape;
                const modelWidth = inputTensorShape[1];
                const modelHeight = inputTensorShape[2];
                //640, 640
                console.log("Model Width", modelWidth, "Model Height", modelHeight);

                const preprocess = (liveFeed, modelWidth, modelHeight) => {
                    let xRatio;
                    let yRatio;
                    const input = tf.tidy(()=>{
                        const webcamFeed = tf.browser.fromPixels(liveFeed);
                        const [h, w] = webcamFeed.shape.slice(0,2);
                        const maxSize = Math.max(w, h);
                        const webCamFeedPadded = webcamFeed.pad([
                            [0, maxSize - h], // padding y [bottom only]
                            [0, maxSize - w], // padding x [right only]
                            [0, 0],
                          ]);
                          xRatio = maxSize/w;
                          yRatio = maxSize/h;
                          return tf.image.resizeBilinear(webCamFeedPadded, [modelWidth, modelHeight]).div(255,0).expandDims(0);
                    })
                    return [input, xRatio, yRatio];
                }
                const [input, xRatio, yRatio] = preprocess(liveFeed, modelWidth, modelHeight);
                console.log("X Ratio", xRatio, "Y Ratio", yRatio);
                const res = model.predict(input);
                const transRes = res.transpose([0,2,1]);
                const boxes = tf.tidy(() => {
                    const w = transRes.slice([0,0,2], [-1,-1,1]); //get width
                    const h = transRes.slice([0,0,3], [-1,-1,1]);//get height
                    const x1 = tf.sub(transRes.slice([0,0,0], [-1,-1,1]), tf.div(w,2));
                    const y1 = tf.sub(transRes.slice([0,0,1], [-1,-1,1]), tf.div(h,2));
                    return tf.concat([y1, x1, tf.add(y1,h), tf.add(x1, w)], 2).squeeze();//y1, x1, y2, x2
                });
                const [scores, classes] = tf.tidy(() => {
                    const rawscores = transRes.slice([0,0,4], [-1,-1, numClass]).squeeze(0);
                    return [rawscores.max(1), rawscores.argMax(1)];
                });
                const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.5);
                const predictionsLength = nms.size;
                console.log("Predictions Length", predictionsLength)
                if (predictionsLength > 0){
                    const boxes_data = boxes.gather(nms, 0).dataSync();
                    const scores_data = scores.gather(nms, 0).dataSync();
                    const classes_data = classes.gather(nms, 0).dataSync();
                    console.log(boxes_data, scores_data, classes_data);
                    const xScale = src.cols/modelWidth;
                    const yScale = src.rows/modelHeight;
                    console.log("Score Data Length", scores_data.length);
                    for (let i=0; i < scores_data.length; ++i){
                        const classID = classes_data[i];
                        console.log(classID)
                        const className = labels[classes_data[i]];
                        const confidenceScore = (scores_data[i] * 100).toFixed(1);
                        let [y1, x1, y2, x2] = boxes_data.slice(i*4, (i+1)*4);
                        x1 *= xRatio * xScale;
                        x2 *= xRatio * xScale;
                        y1 *= yRatio * yScale;
                        y2 *= yRatio * yScale;
                        const width = x2 - x1;
                        const height = y2 - y1;
                        console.log(x1, y1, width, height, className, confidenceScore);
                        let point1 = new cv.Point(x1, y1);
                        let point2 = new cv.Point(x1+ width, y1 + height);
                        cv.rectangle(src, point1, point2, computeColorForLabels(classID), 4);
                        //const text = `${className} - ${Math.round(confidenceScore)/100}`
                        const text = className + " - " + confidenceScore + "%"
                        // Create a hidden canvas element to measure the text size
                        const canvas = document.createElement("canvas");
                        const context = canvas.getContext("2d");
                        context.font = "22px Arial"; // Set the font size and family as needed
                        // Measure the width of the text
                        const textWidth = context.measureText(text).width;
                        cv.rectangle(src, new cv.Point(x1,y1-20), new cv.Point(x1+ textWidth + context.lineWidth, y1), computeColorForLabels(classID),-1)
                        cv.putText(src, text, new cv.Point(x1, y1 - 5),cv.FONT_HERSHEY_TRIPLEX, 0.50, new cv.Scalar(255,255,255,255), 1);
                    }
                    cv.imshow("main-canvas",src);
                }
                else{
                    cv.imshow("main-canvas", src);    
                    }
                // Clear Memory
                tf.dispose(res, transRes, boxes, scores,classes, nms);
                //Call the Function Again After A Delay
                const delay = 1000/24 - (Date.now() - begin);
                setTimeout(predictWebcam, delay)
                //Release the Source Image
                src.delete();

            }
        }
 
 
    }
}
