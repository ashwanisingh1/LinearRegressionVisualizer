let x_vals=[];
let y_vals=[];
let m,b;

const learningRate = 0.4;
const optimizer = tf.train.sgd(learningRate);

function setup()
{
      createCanvas(1400,900);
      m=tf.variable(tf.scalar(random(1)));
      b=tf.variable(tf.scalar(random(1)));
}

function predict(x)
 {
    const xs= tf.tensor1d(x);
    const ys= xs.mul(m).add(b);
    return ys;
 }

function loss(pred,labels)
 {
   return pred.sub(labels).square().mean();
 }

function mousePressed() {
 let x=map(mouseX,0,width,0,1);
 let y=map(mouseY,0,height,1,0);
 x_vals.push(x);
 y_vals.push(y);
}

function draw()
{
  tf.tidy(()=>{
  if(x_vals.length>0)
   {
   const ys=tf.tensor1d(y_vals);
   optimizer.minimize(() => loss(predict(x_vals),ys));
   }
 });
  background(0);
  stroke(255);
  strokeWeight(8);
  for(let i=0;i<x_vals.length;i++)
   {
    let px=map(x_vals[i],0,1,0,width);
    let py=map(y_vals[i],0,1,height,0);
    point(px,py);
   }
   let x1=map(0,0,1,0,width);
   let x2=map(1,0,1,0,width);
   tf.tidy(()=>{
   let ys1=predict([0]);
   let ys2=predict([1]);

   let y1=map(ys1.dataSync(),0,1,height,0);
   let y2=map(ys2.dataSync(),0,1,height,0);
   line(x1,y1,x2,y2);
   const ys=tf.tensor1d(y_vals);
   let l=loss(predict(x_vals),ys).dataSync();
   //console.log(l[0]);


   //let lossy2=map(l[0],0,1,0,height);
   //console.log(lossy2);
   fill(50);
   text('Loss', 10, 30);
   text(l, 50, 30);
   text('Slope', 10, 50);
   text(m.dataSync(), 50, 50);
   });
  //console.log(tf.memory().numTensors);
}
