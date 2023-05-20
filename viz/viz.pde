import java.util.Comparator;

//Input path
String DIR = "../output/demo";

//Press key 'e' to export the results of clusters

//Display
float axisScale = 75.0F; //Can be controlled with keys 'a' and 'd'
float localScale = 0.5F; //Can be controlled with keys 's' and 'w'
float globalScale = 0.2F; //Can be controlled with the mouse wheel
float globalTranslateX = 0.0F; //Can be controlled with mouse dragging
float globalTranslateY = 0.0F; //Can be controlled with mouse dragging

//Similarity
float edgeThreshold = 2.0F; //A threshold that determines the how close can two persons be considered of the same identity. Can be controlled with keys 'j' and 'k'
boolean useEuclidean512 = false; //Whether the Euclidean distance calculated with all 512 dimensions or the two dimensions calculated by t-SNE should be used for identifying the faces. Cannot be controlled inside the programme.
float edgeStep = 0.05F; //The step value for controlling edge threshold

//Image size normalizing
boolean normalize_size = true; //Whether the face images' size should be normalised
int norm_h = 200; //The target size for normalizing

//Variables
int pMouseX = -1, pMouseY = -1;
HashMap<String, Face> faces;
ArrayList<Edge> edges;
Table source, tsne;

class Face {
  float x, y;
  PImage image;
  int frame, face;
  String id;
  int w, h;
  float[] embedding;
  String box;
  ArrayList<Face> connectedFaces;
  
  public Face(float x, float y, PImage image, int frame, int face, String box) {
    this.x = x;
    this.y = y;
    this.image = image;
    this.frame = frame;
    this.face = face;
    this.id = "t" + frame + "f" + face;
    this.w = image.width;
    this.h = image.height;
    this.embedding = new float[0];
    
    this.box = box;
  }
  
  public Face(float x, float y, PImage image, int frame, int face, String box, float scale) {
    this.x = x;
    this.y = y;
    this.image = image;
    this.frame = frame;
    this.face = face;
    this.id = "t" + frame + "f" + face;
    this.w = (int)(image.width * scale);
    this.h = (int)(image.height * scale);
    this.embedding = new float[0];
    
    this.box = box;
  }
  
  public void setPosition(float x, float y) {
    this.x = x;
    this.y = y;
  }
  
}

class Edge {
  Face a, b;
  float dis;
  
  public Edge(Face a, Face b, float dis) {
    this.dis = dis;
    this.a = a;
    this.b = b;
  }
}

void setup() {
  size(960, 640, P2D);
  faces = new HashMap<String, Face>();
  source = loadTable(DIR + "/embeddings.tsv", "tsv");
  tsne = loadTable(DIR + "/tsne.csv", "csv");
  
  for (TableRow row : source.rows()) {
    int frame = row.getInt(1);
    int face = row.getInt(2);
    String box = row.getString(3);
    String id = "t" + frame + "f" + face;
    String imgPath = DIR + "/" + id + ".jpg";
    
    PImage img = loadImage(imgPath);
    if(!normalize_size) {
      faces.put(id, new Face(0.0f, 0.0f, img, frame, face, box));
    }else{
      float scale = (float)norm_h / (float)img.height;
      faces.put(id, new Face(0.0f, 0.0f, img, frame, face, box, scale));
    }
    if(useEuclidean512) {
      String s = row.getString(4);
      String[] embedding = s.substring(1, s.length()-1).split(", ");
      float[] eu = new float[embedding.length];
      for(int i=0; i<embedding.length; i++) {
        eu[i] = Float.parseFloat(embedding[i]);
      }
      faces.get(id).embedding = eu;
    }
  }
  
  for(TableRow row : tsne.rows()) {
    float x = row.getFloat(2);
    float y = row.getFloat(3);
    String id = "t" + row.getString(0) + "f" + row.getString(1);
    
    faces.get(id).setPosition(x, y);
  }
  
  println(faces.size() + " faces found in the results.");
  
  float minDis = 9999;
  float maxDis = -9999;
  edges = new ArrayList<Edge>();
  ArrayList<String> storedEdges = new ArrayList<String>();
  for(Face a : faces.values()) {
    for(Face b : faces.values()) {
      if(a == b) continue;
      String test = b.id + "-" + a.id;
      if(storedEdges.contains(test)) continue; //If the revered edge of this exists, skip
      float dis = 0;
      if(useEuclidean512) {
        dis = 0;
        for(int i=0; i<512; i++) {
          dis += sq(a.embedding[i] - b.embedding[i]);
        }
        dis = sqrt(dis);
      }else{
        dis = sqrt(sq(a.x - b.x) + sq(a.y - b.y));
      }
      edges.add(new Edge(a, b, dis));
      if(dis > maxDis) maxDis = dis;
      if(dis < minDis) minDis = dis;
      storedEdges.add(a.id + "-" + b.id);
    }
  }
  storedEdges.clear();
  
  if(useEuclidean512) {
    edgeThreshold = 15.0f;
    edgeStep = 0.01f;
  }
  
  println("Maximum distance: " + maxDis);
  println("Minimum distance: " + minDis);
}

void draw() {
  background(196);
  imageMode(CENTER);
  
  if (keyPressed) {
    if (key == 'r') {
      globalTranslateX = globalTranslateY = 0;
      globalScale = 1;
    }else if(key == 'w') {
      axisScale += 0.5f;
    }else if(key == 's') {
      axisScale -= 0.5f;
    }else if(key == 'a') {
      localScale += 0.01f;
    }else if(key == 'd') {
      localScale -= 0.01f;
    }else if(key == 'k') {
      edgeThreshold += edgeStep;
    }else if(key == 'j') {
      edgeThreshold -= edgeStep;
    }
  }
  
  translate((float)width/2, (float)height/2);
  scale(globalScale);
  translate(globalTranslateX, globalTranslateY);
  stroke(1);
  
  for(Face f : faces.values()) {
    push();
      translate(f.x * axisScale, f.y * axisScale);
      scale(localScale);
      image(f.image, f.x, f.y, f.w, f.h);
    pop();
  }
  
  push();
    scale(axisScale);
    stroke(0, 64);
    strokeWeight(0.05);
    for(Edge e : edges) {
      if(e.dis < edgeThreshold)
        line(e.a.x, e.a.y, e.b.x, e.b.y);
    }
  pop();
}

void mouseWheel(MouseEvent event) {
  float e = event.getCount();
  float scale = map(e, -50.0f, 50.0f, -1.0f, 1.0f);
  globalScale -= scale;
  if(globalScale <= 0.05f) globalScale = 0.05f;
}

void mousePressed() {
  pMouseX = mouseX;
  pMouseY = mouseY;
}

void mouseDragged() {
  float xMov = mouseX - pMouseX;
  float yMov = mouseY - pMouseY;
  
  globalTranslateX += map(xMov, -10, 10, -50, 50);
  globalTranslateY += map(yMov, -10, 10, -50, 50);
  
  pMouseX = mouseX;
  pMouseY = mouseY;
}

void keyReleased() {
  if(key == 'e') {
    exportClusters();
  }
}

int faceCounter = 0;
ArrayList<Face> unclusteredFaces;

void exportClusters() {
  for(Face f : faces.values()) {
    f.connectedFaces = new ArrayList<Face>();
  }
  for(Edge e : edges) {
    //if(e.a.connectedFaces.contains(e.b)) {println("WARNING");}
    //if(e.b.connectedFaces.contains(e.a)) {println("WARNING");}
    if(e.dis < edgeThreshold) { 
      e.a.connectedFaces.add(e.b);
      e.b.connectedFaces.add(e.a);
    }
  }
  
  faceCounter = 0;
  unclusteredFaces = new ArrayList<Face>(faces.values());
  ArrayList<ArrayList<Face>> clusters = new ArrayList<ArrayList<Face>>();
  Face[] all = faces.values().toArray(new Face[faces.size()]);
  
  for(Face f : all) {
    if(unclusteredFaces.contains(f)) {
      ArrayList<Face> newCluster = new ArrayList<Face>();
      for(Face f2 : all) {
        if(f2 == f) continue;
        if(!unclusteredFaces.contains(f2)) {
          continue;
        }
        if(isConnected(f, f2) && !newCluster.contains(f2)) {
          newCluster.add(f2);
          unclusteredFaces.remove(f2);
          faceCounter++;
        }
      }
      clusters.add(newCluster);
    }
  }
  
  clusters.sort(Comparator.comparing(ArrayList<Face>::size).reversed());
  
  Table table = new Table();
  table.addColumn("cluster");
  table.addColumn("frame");
  table.addColumn("face");
  table.addColumn("box");
  
  int c = 0;
  for(ArrayList<Face> cluster : clusters) {
    for(Face f : cluster) {
        TableRow newRow = table.addRow();
        newRow.setInt("cluster", c);
        newRow.setInt("frame", f.frame);
        newRow.setInt("face", f.face);
        newRow.setString("box", f.box);
    }
    c++;
  }
  String output = DIR + "/clusters.tsv";
  saveTable(table, output, "tsv");
  
  println();
  println(faceCounter + " faces in " + clusters.size() + " clusters exported at " + output);
}

boolean isConnected(Face a, Face b) {
  ArrayList<Face> toDo = new ArrayList<Face>();
  ArrayList<Face> done = new ArrayList<Face>();
  done.add(a);
  toDo.add(a);
  
  while(toDo.size() > 0) {
    Face obj = toDo.get(0);
    toDo.remove(obj);
    
    if(obj.connectedFaces.contains(b)) {
      return true;
    }
    
    for(Face newFace : obj.connectedFaces) {
      if(!done.contains(newFace)) {
        toDo.add(newFace);
        done.add(newFace);
      }
    }
  }
  return false;
}
