# Photometric Stereo

## Mục tiêu và vấn đề

Từ bài trước ta có thê biết được từ 1 ảnh duy nhất ta không thể tái tạo 3D bằng phương pháp photometric stereo do từ 1 image intensity có thể ánh xạ ra vô vàn điểm thuộc cùng 1 iso-brighness conner, không thể tính chính xác giá trị p,q tại trên reflectance map.

Photometric stereo: sử dụng nhiều ảnh của vật thể được chụp từ cùng 1 camera, camera đặt cùng 1 vị trí, nhiều ảnh chụp ở các điều kiện chiếu sáng khác nhau (nhiều nguồn sáng khác nhau). Đây là phương pháp được phát triển bởi Robert Woodham và được sử dụng rộng rãi đặc biệt là trong các môi trường chúng ta đã biết cấu trúc của nó.

Tuy nhiên trong quá trình chụp, người chụp có thể không đưng yên mà có thể chuyển động 1 ít nên ảnh có thể lệch vài pixel. Phỉa xử lý thủ công để dịch tất cả các ảnh về 1 đieẻm chung rồi mới áp dụng phương pháp này (đồ án)

## Ý tưởng photometric stereo

Thực tế việc sử dụng nhiều ảnh có nghĩa là ta sử dụng nhiều gái trị I trong mỗi pixel từ đó ta có thể thu được nhiều phương trình p, q và có thể giải được nó lấy được giá trị thông tin p, q chính xác và tính được surface orientation.

![](images/10.%20Photometric%20stereo.png)

Giả sử ta có thiết lập như hình trên bên phải, camera đứng nguyên và nó đứng rất xa nhên hướng của nó tương tự với mọi điểm trên bề mặt. Tương tự nguồn sáng s cũng rất xa nên mọi điểm trên surface cùng nhìn nguồn với 1 hướng s giống nhau. Ta thực hiện chụp 3 bức ảnh với 3 nguồn sáng S1, S2, S3 khác nhau.

Ký hiệu các si, Reflectance map Ri và các image intensity Ii.

![](images/10.%20Basic%20Ideal.png)

Giả sự I1 = 0, 9. Ta sẽ biết được surface normal map nằm trong đường nào của Reflectance map.
Tương tự I2 = 0.7. Ta sẽ biết được surface normal map nằm trong đường nào của Refelectance map. Giao 2 đường lại ta tìm ra được 2 điểm có khả năng là surface normal => Tương tự với I3 ta sẽ tìm ra được điểm duy nhất biểu diễn surface normal.

## Tổng kết

![](images/9.%20Reflectance%20Map%20Lambertian%203.png)

Để tìm được surface normal map, ta phải có ít nhất k bức ảnh với k nguồn sáng khác nhau và camera chụp phải giống nhau. Phỉa biết được các tính chất phản xạ của bề mặt như BRDF để tái tạo được reflectance map với mỗi ảnh.

Với Lambertian Surface cần ít nhất là 3 images. Thực tế nếu có nhiều hình ảnh ta có thể đi xa hơn tính được albedo (hiệu suất phản xạ) tại mỗi điểm.

Tuy nhiên nếu có bất kỳ pixel nào bị thiếu trong 3 images thì ta không thể tái tạo 3d được, do đó ta cần đặt nguồn sáng sao cho nó vừa khác nhau, vừa phải làm cho image thấy được mọi điểm trên surface. Và thậm chí nếu như BRDF phức tạp như ảnh dị hướng (anisotropic) và trường hợp tệ nhất alf specular perfect (gương hoàn hảo). Đối với bất kỳ nguồn sáng điểm nào (point light source) sẽ chỉ có 1 hướng nhìn duy nhất mà bạn có được tia phản xạ (tức nơi đặt camera) => Cần vô hạn nguồn sáng để tái tạo tất cả pháp tuyến bề mặt với perfect mirror.