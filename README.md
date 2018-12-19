# SimpleNN
A simple neural network implemented with pure c++, trained with ADAM optimizer, test with MNIST dataset (98.31% accuracy)

## Compile
```sh
make
./MnistTrainer mnist.nn #train neural network (multithreaded)
./MnistTest mnist.nn    #or bests/784-200-200-10_0.nn for the best result in my computer
./MnistUI bests/784-200-200-10_width_98_31.nn #a ui to recognize handwritten digit (written with SFML)
```

## MnistUI
### Usage
drag to write a digit, press SPACE to recognize digit with neural network, and the result will be output in console
## Screenshots
![](https://raw.githubusercontent.com/AdamYuan/SimpleNN/master/screenshots/1.png)
```
            .?????..        
          .?@@@@@@@@@??.    
         .?@??....???@@?.   
        .?@@.       .?@??   
        ?@??.       .?@??   
        ?@?.        .@@?.   
        ?@??.      .?@@.    
        .?@@.     .?@@?.    
         .?@??.  .?@@?.     
          .?@@?.?@@@?.      
           .?@@@@@?..       
           .?@@@@@?.        
          ?@@@@@@@@?.       
       .?@@@?.. .??@?.      
     .??@@?..    ..?@??.    
    .?@@?.         .?@@.    
   ?@@??            .?@?.   
 .??@?.              .?@?.  
 .@@?.               .?@@.  
 ?@?.                 .@@?  
 ?@?.                 .?@?  
 ?@@?.                .@@.  
 .??@?..             .?@@.  
  .??@@?.           .?@??   
    .?@@@?..    ...?@@@.    
      .??@@@@@@@@@@@??.     
        ..?????????..       
                            
recognize: 8
nn output: 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.014501 0.000000 
```
![](https://raw.githubusercontent.com/AdamYuan/SimpleNN/master/screenshots/2.png)
```
                    .?      
       ...         .?@.     
       ???         .?@.     
      .???         .?@.     
      .???         .?@.     
      .@?.         .?@.     
      ?@?.         .?@.     
     .?@.          .?@.     
     ??@.          .?@.     
    .@@?.          .?@? ... 
    .@?.            ?@?.???.
   .?@?            .?@@@@@?.
   ?@?.          .?@@@@@?.  
 .?@@.     ..???@@@@@@?.    
 ?@@@....???????????@@?     
.?@@@??@@@@???..   .?@?     
?@@@@@???..         ?@?     
                    ?@?     
                   .?@?     
                   .?@.     
                   .?@.     
                   .?@.     
                   .?@.     
                   .?@.     
                   .?@.     
                   .?@.     
                   .??.     
                    ..      
recognize: 4
nn output: 0.000000 0.000000 0.000000 0.000000 0.999000 0.000000 0.000000 0.000000 0.000000 0.000000 
```
