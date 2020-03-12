        program calc_hoh_pot
        implicit real*8 (a-h,o-z)
        include 'param.inc'
        dimension x(3,3),rij(nwalk,3),v(nwalk)
        open (unit=8,file='hoh_coord.dat',status='old')
        open (unit=9,file='hoh_pot.dat',status='unknown')
        read(8,*)np
        do k = 1,np
           read(8,*)((x(j,i),j=1,3),i=1,3)
           r1 = 0.
           r2 = 0.
           ct = 0.
           do j = 1,3
              d1 = x(j,3)-x(j,1)
              d2 = x(j,3)-x(j,2)
              r1 = r1 + d1**2
              r2 = r2 + d2**2
              ct = ct + d1*d2
           enddo
           rij(k,1) = sqrt(r1)
           rij(k,2) = sqrt(r2)
           rij(k,3) = acos(CT/rij(k,1)/rij(k,2))
C          rij(k,3) = acos(CT/r1/r2)

        enddo
        call vibpot(rij,v,np)
        do i = 1,np
           write(9,*)v(i)
        enddo
        stop
        end
        
