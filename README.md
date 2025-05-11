# Brain-Tumour-CNN-YOLOv2-Medical-SAM
Upgraded version of previous "Brain-Tumour-CNN" project.
Kurulum:
- zip dosyanını "buraya çıkart" diyerek çıkartın.
- matlab'i açın.
- klasördeki googlenet isimli dosyaya çift tıklayın.
- matlab'de add-on manager'ın açılması lazım.
- googlenet'in kurulumunu sağlayın.
- alexnet, resnet50, medicalsegmentanythingmodel için aynı adımları tekrarlayın.
- matlab'de Deep Learning Toolbox, Image Processing Toolbox, Computer Vision Toolbux ve Medical Imaging Toolbox'un kurulu olduğundan emin olun.
- app isimli dosyayı app designer'dan açın.
Kullanım:
- sol kısımda "TRAIN" kısmı bulunmaktadır.
- istediğiniz parametreleri seçerek model trainleyin.
- map değerleri ve f1 score, recall ve precision değerleri yazdırılır; model kaydedilir.
- sağ kısımda "PREDICT" kısmı bulunmaktadır.
- sol kısımı kullanarak trainlediğiniz modeller model kısmında seçilebilir şekilde listelenir.
- istediğiniz parametreleri ve görseli seçerek prediction yapın.
- sonuç olarak tümör yoksa "Clean", varsa "Has Tumour" yazılmaktadır.
- ayrıca tümör bulunursa tümörün yerini belirten işaretler çizdirilir.
