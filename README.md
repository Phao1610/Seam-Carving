# Seam-Carving-
- Seam Carving is a new way to crop images without losing important content in the image. This is often referred to as "content-aware" cropping or image retargeting. 
- During this period, the group performed the following tasks: 
    First create a new image (mask) by clipping a bouding box in the selected image. 
    
    Then save the position of the mask in the original image to the list (lst_roi).
    
    ![image](https://user-images.githubusercontent.com/106755542/171824069-76184d9d-e3d8-4a0c-b2a8-d89d59e3f0aa.png)
    
    Next, we will create an Energy map. The first step is to calculate the energy value for each pixel. The paper identifies many different energy functions that can be used and the team decided to use the most basic formula.

    ![image](https://user-images.githubusercontent.com/106755542/171824307-6e8070a6-127a-4ff7-ade1-de29f2b1dbd0.png)

    I here is the image, this equation tells us that, for each pixel in the image, for every channel, we do the following:
      
      Find partial derivatives along the x . axis
      
      Find partial derivatives along the y trục axis
      
      Sum their absolute values
    
    Use the Sobel operator to calculate the derivative. This is a compound kernel that is run on the image on every channel. Here is the filter in two different directions of the image:
    
    ![image](https://user-images.githubusercontent.com/106755542/171824574-748760e4-2f4a-4a8d-8fb1-811f7c361907.png)
    
    The above method is the same as edge detection method.
    
    ![image](https://user-images.githubusercontent.com/106755542/171824885-c1f5797f-4469-4918-8f51-7438a11495d7.png)
    
    Next, we will find the cloud lines that go from the top to the bottom of the image and have the lowest energy, but these cloud lines must contain a mask.
    
    This line must be connected 8: this means that every pixel in the line must touch the next pixel in the line through an edge or corner.
    
    The idea of our group is to share any cloud lines that pass through the mask (M[i,j]), I will assign it a very small price so that when calculating the total energy of those lines it will be the smallest and delete it. Go.
    
    ![image](https://user-images.githubusercontent.com/106755542/171825101-6a4466f6-ddb5-4252-a28c-3f7ae64eabd8.png)
    
    To find the cloud path, we use the following formula:
    
      M[i,j]=e[i,j]+min⁡(M[i-1,j-1],M[i-1,j],M(i-1,j+1])
    
    To understand more about the formula in, you can see this picture:
    
    ![image](https://user-images.githubusercontent.com/106755542/171825271-e429b7ff-e3c3-4d32-9892-36abbd42746d.png)
    
    Lets make a 2D array call, M to store the minimum energy value seen up to that pixel and this minimum energy must contain the mask. Thus, the minimum energy required to propagate from the top of the image to the bottom will be present in the row from the top to the bottom of the mask in M's image. It is necessary to trace this back to find the list of pixels. included in this junction, so we'll capture the values with a call to the 2D backtrack array.
    
    Next is to remove that cloud line from the image as described above. After deleting, you should reshape the image.
    
    Repeat the above steps until the deletion is complete. The condition to know that the deletion has been completed is:
    
      The variable lst_roi[2] is the column top position of the mask in the image
      
      The variable lst_roi[3] is the position at the end of the column of the mask in the image
      
      We repeat lst_roi[3] – lst_roi[2] times, corresponding to deleting lst_roi[3] – lst_roi[2] which way. That also means that the mask has been removed from the original image.
    
    It is also possible to use this method that delete rows but have to rotate the image by an angle of 90^0 before deleting.
    
    ![image](https://user-images.githubusercontent.com/106755542/171825609-aaa8eac0-cbbc-4186-a200-6693bd205c9b.png)
    
    We all see that it will not erase the cloud lines related to the surrounding aircraft.

 
