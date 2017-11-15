library(ggplot2)
library(RColorBrewer)

myCol = brewer.pal(4,"Blues")[2:4]
myPink <- "#FEE0D2"
myRed <- "#99000D"

z2<- ggplot(mtcars,aes(x=wt,y=mog,col=factor(cyl)))+
  geom_point(size =2, alpha=0.8)+
  geom_smooth(method = "lm",se = F)+
  facet_grid(.~cyl)+
  scale_color_manual('Cylinders',values = myCol)+
  
  scale_y_continuous("Miles/(US) gallon")+
  scale_x_continuous("Weight (lb/1000)")

theme_pink<-theme(panel.background = element_blank(),
                  legend.key = element_blank(),
                  strip.background = element_blank(),
                  plot.background = element_rect(fill = myPink,color = 
                                                   "black",size=3),
                  panel.grid = element_blank(),
                  axis.line = element_line(color = "black"),
                  axis.ticks = element_line(color = "black"),
                  strip.text = element_text(size=16, color = myRed),
                  axis.title.x = element_text(color = myRed,hjust = 0,face = "italic"),
                  axis.text = element_text(color="black"),
                  legend.position = "none")

theme_update(panel.background=element_blank(),
             legend.key = element_blank(),
             legend.background = element_blank(),
             strip.background = element_blank(),
             )
