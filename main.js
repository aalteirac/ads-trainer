import {KNNImageClassifier} from 'deeplearn-knn-image-classifier';//
import * as dl from 'deeplearn';//
import {Tensor} from 'deeplearn';//
import {IndexedDB} from './indexedDB.js';

const NB_CLASSIFS = 2;
const IMAGE_SIZE = 227;

const TOPK = 5;


class Main {
  constructor(){
    this.infoTexts = [];
    this.inputTexts = [];
    this.buttons = [];
    this.training = -1;
    this.videoPlaying = true;
    this.tm;
    this.curIndex;
    this.knn = new KNNImageClassifier(NB_CLASSIFS, TOPK);

    document.getElementById("export").addEventListener('click', () => {
      this.exportModel();
    })

    document.getElementById("save").addEventListener('click', () => {
      this.save("trainedSampleCount", this.knn.classExampleCount)
      this.save("trained", this.knn.classLogitsMatrices)
      for(let i=0;i<this.knn.classLogitsMatrices.length; i++){
        this.save("trainedValue"+i, this.knn.classLogitsMatrices[i]!=null?this.knn.classLogitsMatrices[i].dataSync():null)
        this.save("trainedValueLabel"+i, this.inputTexts[i].value)
      }
      alert("saved !");
    })

    document.getElementById("restore").addEventListener('click', () => {
      this.getVal("trainedSampleCount").then((res)=>{
        var prom=[]
        for(let i=0;i<res.length; i++){
          if(res[i]>0)
            prom.push(this.getVal("trainedValue"+i));
        }
        Promise.all(prom).then((values)=>{
          this.getVal("trained").then((vl)=>{
            values.map((el,id)=>{
              this.knn.classLogitsMatrices[id]=new Tensor (vl[id].shape, vl[id].dtype,Object.values(el), vl[id].dataId);
              this.getVal("trainedValueLabel"+id).then((lab)=>{
                this.inputTexts[id].value=lab;
              })

            })
            this.knn.classExampleCount=  res;
          })
        });
      })

      //this.getVal("trained").then((sv)=>{
      //  this.getVal("trainedValue0").then((val0)=>{
      //    this.getVal("trainedValue1").then((val1)=>{
      //      this.getVal("trainedSampleCount").then((res)=>{
      //        this.knn.classLogitsMatrices[0]=new Tensor (sv[0].shape, sv[0].dtype,Object.values(val0), sv[0].dataId);
      //        this.knn.classLogitsMatrices[1]=new Tensor (sv[1].shape, sv[1].dtype,Object.values(val1), sv[1].dataId);
      //        this.knn.classExampleCount=  res;
      //        alert("restored");
      //      })
      //    })
      //  })
      //})
    })


    // Create training buttons and info texts    
    for(let i=0;i<NB_CLASSIFS; i++){
      const div = document.createElement('div');
      document.getElementById("traincol").appendChild(div);
      div.setAttribute("class","form-inline")
      //div.style.marginBottom = '10px';

      // Create training button
      const button = document.createElement('button')
      button.innerText = "Train "+i;
      button.setAttribute("disabled", "");
      button.setAttribute("type", "button");
      button.setAttribute("class", "btn btn-primary");
      div.appendChild(button);
      const input = document.createElement('input')
      input.setAttribute("type", "text");
      input.setAttribute("class", "form-control");
      input.setAttribute('placeholder',"Thing to recognize  "+i);
      div.appendChild(input);
      // Listen for mouse events when clicking the button
      button.addEventListener('click', () => {
        if(this.training === -1)
          this.training = i;
        else{
          this.training = -1
        }
        this.disableAll();
        document.getElementById("predict") .innerText="...LEARNING...";
        setTimeout(()=>{
          this.training = -1;
          this.enableAll();
        },12000)
      });

      const infoText = document.createElement('span')
      infoText.innerText = " 0 sample";
      div.appendChild(infoText);
      this.buttons.push(button);
      this.infoTexts.push(infoText);
      this.inputTexts.push(input);
    }

    //this.initCam(true)
    //this.initStream();
    // Load knn model
    //this.knn.
    this.knn.load()
        .then(() => {
          this.start()
          this.enableAll();
          document.getElementById("predict").innerText= "Ready to be trained";
        });
  }

  disableAll(){
    this.buttons.map((bt)=>{
      bt.setAttribute("disabled","")
    })
    document.getElementById("restore").setAttribute("disabled","");
  }
  enableAll(){
    this.buttons.map((bt)=>{
      bt.removeAttribute("disabled")
    })
    this.getVal("trainedSampleCount").then((res)=> {
        document.getElementById("restore").removeAttribute("disabled");
    })
  }

  getFileModel(fName){
    return new Promise((resolve, reject)=>{
      var req = new XMLHttpRequest();
      req.open("GET", "model/"+fName, true);
      req.onreadystatechange = function() {
        if (req.readyState === 4) {
          if (req.status === 200) {
            var allText = req.responseText;
            resolve(allText)
          }
        }
      }
      req.send(null);
    })

  }

  httpGet(theUrl) {
    return new Promise((resolve, reject)=> {
      var xmlHttp = new XMLHttpRequest();
      xmlHttp.onreadystatechange = function () {
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
          resolve(xmlHttp.responseText);
      }
      xmlHttp.onerror = function () {
        reject("ko");
      };
      xmlHttp.open("GET", theUrl, true);
      xmlHttp.setRequestHeader('Access-Control-Allow-Origin', '*')
      xmlHttp.send(null);
    })
  }

  createDLLink(obj,label){
    var blob = new Blob([ JSON.stringify(obj) ], {
      type : "text/plain;charset=utf-8;"
    });
    var modUrl = URL.createObjectURL(blob);

    //var data = "text/plain;charset=utf-8," + encodeURIComponent(JSON.stringify(obj));
    var a = document.createElement('a');
    a.href = modUrl;
    a.download = label+'_data.json';
    a.innerHTML = `download ${label}`;
    var container = document.getElementById('container');
    container.appendChild(a);
    container.appendChild(document.createElement('br'));
  }

  save(nm,val){
    var indexed=new IndexedDB();
    indexed.openDB().then((r)=>{
      indexed.store(nm,val);
    },(t)=>{
      console.log("KO",t);
    });
  }

  getVal(nm){
    var indexed=new IndexedDB();
    return indexed.openDB().then((r)=>{
      return indexed.getValue(nm);
    },(t)=>{
      console.log("KO",t);
    });
  }

  exportModel(){
    this.createDLLink(this.knn.classLogitsMatrices, "Classifiers_matrices");
    this.createDLLink(this.knn.classExampleCount, "Classifier_Sampling");
    this.createDLLink(this.knn.classLogitsMatrices[0].dataSync(), "Classifier_1_values");
    this.createDLLink(this.knn.classLogitsMatrices[1].dataSync(), "Classifier_2_values");
  }

  initStream(){
 
    this.video.src="http://192.168.1.20/df"
    this.video.width = IMAGE_SIZE;
    this.video.height = IMAGE_SIZE;
    this.video.addEventListener('playing', ()=> this.videoPlaying = true);
    this.video.addEventListener('ended', ()=> {
      setTimeout(() => {
        this.video.src="http://192.168.1.20/df"
        this.video.play();
      }, 5000);
      
    });
    
    this.start();
  }
  initCam(ck){
    this.stop();
    var cm=ck?"user":"environment"
    const constraints = {
      advanced: [{
        facingMode: cm
      }]
    };
    navigator.mediaDevices.getUserMedia({video: constraints, audio: false})
        .then((stream) => {
          this.video.srcObject = stream;
          this.video.width = IMAGE_SIZE;
          this.video.height = IMAGE_SIZE;

          this.video.addEventListener('playing', ()=> this.videoPlaying = true);
          this.video.addEventListener('paused', ()=> this.videoPlaying = false);
          this.start();
        })

  }

  start(){
    if (this.timer) {
      this.stop();
    }
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }

  stop(){
    this.video.pause();
    cancelAnimationFrame(this.timer);
  }

  checkStability(id,text){
    //this.curIndex=id;
    if(this.curIndex!=id) {
      this.tm=new Date().valueOf();
      this.curIndex=id
    }
    else {
      var dd=new Date().valueOf()-this.tm;
      if(dd>3000){
        this.tm=new Date().valueOf();
        //console.log(text);
        document.getElementById("predict") .innerText= text;
        //this.httpGet("http://localhost:8099/message?mess="+encodeURI(text))
      }
    }
    return 
  }
  animate(){
    if(this.videoPlaying){
      var image;
      var canvas = document.createElement('canvas');
      var context = canvas.getContext('2d');
      var img = document.getElementById('lv');
      img.crossOrigin="Anonymous";
      canvas.width = img.width;
      canvas.height = img.height;
      context.drawImage(img, 0, 0 );
      var myData = context.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
      const image = dl.fromPixels(myData);
      if(this.training != -1){
        //if(this.knn.classExampleCount[this.training]>=30){
        //  this.training=-1;
        //}
        //else
          this.knn.addImage(image, this.training);


      }

      const exampleCount = this.knn.getClassExampleCount();
      if(Math.max(...exampleCount) > 0){
        document.getElementById("save").removeAttribute("disabled");
        this.knn.predictClass(image)
            .then((res)=>{
              for(let i=0;i<NB_CLASSIFS; i++){
                if(res.classIndex == i  && this.training==-1){
                  this.checkStability(res.classIndex,this.inputTexts[i].value);
                }
                if(exampleCount[i] > 0){
                  this.infoTexts[i].innerText = ` ${exampleCount[i]} samples - ${res.confidences[i].toFixed(2)*100}%`
                }
              }
            })
            .then(()=> image.dispose())
      } else {
        image.dispose()
      }
    }
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }
}


var mymain;
window.addEventListener('load', () => mymain=new Main());












