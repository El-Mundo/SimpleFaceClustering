String DIR = "../output/我们村里的年轻人（上集）";
HashMap<String, Face> faces;
ArrayList<Edge> edges;
Table source, tsne;

float axisScale = 75.0F;
float localScale = 0.5F;
float globalScale = 0.2F;
float globalTranslateX = 0.0F;
float globalTranslateY = 0.0F;

float edgeThreshold = 2.0F;

boolean normalize_size = true;
int norm_h = 200;

int pMouseX = -1, pMouseY = -1;

class Face {
  float x, y;
  PImage image;
  int frame, face;
  String id;
  int w, h;
  
  public Face(float x, float y, PImage image, int frame, int face) {
    this.x = x;
    this.y = y;
    this.image = image;
    this.frame = frame;
    this.face = face;
    this.id = "t" + frame + "f" + face;
    this.w = image.width;
    this.h = image.height;
  }
  
  public Face(float x, float y, PImage image, int frame, int face, float scale) {
    this.x = x;
    this.y = y;
    this.image = image;
    this.frame = frame;
    this.face = face;
    this.id = "t" + frame + "f" + face;
    this.w = (int)(image.width * scale);
    this.h = (int)(image.height * scale);
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
    String id = "t" + frame + "f" + face;
    String imgPath = DIR + "/" + id + ".jpg";
    String embedding = row.getString(4);
    
    PImage img = loadImage(imgPath);
    if(!normalize_size) {
      faces.put(id, new Face(0.0f, 0.0f, img, frame, face));
    }else{
      float scale = (float)norm_h / (float)img.height;
      faces.put(id, new Face(0.0f, 0.0f, img, frame, face, scale));
    }
  }
  
  for(TableRow row : tsne.rows()) {
    float x = row.getFloat(2);
    float y = row.getFloat(3);
    String id = "t" + row.getString(0) + "f" + row.getString(1);
    
    faces.get(id).setPosition(x, y);
  }
  
  println(faces.size());
  
  float minDis = 9999;
  float maxDis = -9999;
  edges = new ArrayList<Edge>();
  for(Face a : faces.values()) {
    for(Face b : faces.values()) {
      if(a == b) continue;
      float dis = sqrt(sq(a.x - b.x) + sq(a.y - b.y));
      edges.add(new Edge(a, b, dis));
      if(dis > maxDis) maxDis = dis;
      if(dis < minDis) minDis = dis;
    }
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
    }else if(key == 'j') {
      edgeThreshold += 0.01f;
    }else if(key == 'k') {
      edgeThreshold -= 0.01f;
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
